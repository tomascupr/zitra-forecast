"""Zitra-Titan-v3 (100% Pure Apache-2.0): Fast 4-Architecture Dynamic-32K Probabilistic Forecaster.

Bundles four genuinely orthogonal, 100% Apache-2.0 foundation architectures into a single
unified safetensors checkpoint (~1.06B total params, ~2.12 GB bf16, 0 learned router weights,
zero benchmark overlap):
  0. Zitra-t0-256M-Dyn32K    (255.6M, Patch-Decoder Transformer — Apache-2.0, dynamic RoPE 512/1536/2048 up to 32K)
  1. Zitra-Chronos2-120M     (119.5M, T5 Encoder-Only          — Apache-2.0, Head-Preserving FullLoRA WiSE-FT)
  2. Zitra-PatchTST-385M     (384.6M, 30-Block Conformer       — Apache-2.0, Head-Preserving 180-linear FullLoRA)
  3. Zitra-iAmTime-302M      (302.1M, T5 Enc-Dec 4-Expert MoE  — Apache-2.0, Head-Preserving Encoder FullLoRA WiSE-FT)

Engineering & Routing (Pop-Quiz v3 Fast):
  - Parallel CUDA streams across independent experts (t0, Chronos-2, PatchTST, iAmTime)
  - Precision-adaptive autocast: fp32 TF32 on short / high-offset cumulative series, bf16 FlashAttention on long series
  - Strictly causal Pop-Quiz 8K vs 16K context gate on long known-covariate series
  - Fast backtests: symmetric_averaging=True on main forecast, False on internal routing cuts + fused pass batching
  - Optional `fast=True` mode (skips backtests for ~50 ms single-pass serving)
  - Dynamic RoPE scale_base (512 for T<=8192, 1536 for T<=16384, 2048 for T>16384)
  - Outlier-proof median-centered log-loss Bayesian posterior with horizon-matched w_cap (up to 192)
  - Independent per-expert Causal Pop-Quiz Self-Gated Cross-Inference (cross_learning="auto", T<=1024, V==1, B>1, no_cov)
  - Exact piecewise-linear Linear-Pool Mixture-CDF inversion embedded cleanly into 21 quantiles.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
import json
from pathlib import Path
import types
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from transformers import PreTrainedModel, T5Config
from transformers.models.t5.modeling_t5 import T5LayerNorm
from t0 import MaskType, T0Forecaster, TimeSeries, VariateType, batch_series
from chronos.chronos2 import Chronos2CoreConfig, Chronos2Model, Chronos2Pipeline
from tsfm_public.models.patchtst_fm import PatchTSTFMConfig, PatchTSTFMForPrediction

try:
    from ._iamtime import RoPE, iAmTimeModel, iAmTimePipeline
except ImportError:
    from _iamtime import RoPE, iAmTimeModel, iAmTimePipeline

DECILES = [i / 10.0 for i in range(1, 10)]
QUANTILES_21 = [
    0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
    0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99,
]
DECILE_IDX_21 = [2, 4, 6, 8, 10, 12, 14, 16, 18]
EXPERT_NAMES = ("t0", "chronos2", "patchtst", "iamtime")


@dataclass
class ZitraV3ForecastOutput:
    quantiles: torch.Tensor
    quantile_levels: list[float]


def _as_context(context, observation_mask=None, device="cuda"):
    x = context if isinstance(context, torch.Tensor) else torch.as_tensor(context, dtype=torch.float32)
    x = x.to(device=device, dtype=torch.float32)
    if x.ndim == 1:
        x = x[None, None, :]
    elif x.ndim == 2:
        x = x[:, None, :]
    elif x.ndim != 3:
        raise ValueError(f"Expected 1D/2D/3D context, got {tuple(x.shape)}")
    if observation_mask is not None:
        m = observation_mask if isinstance(observation_mask, torch.Tensor) else torch.as_tensor(observation_mask)
        m = m.to(device=device, dtype=torch.bool)
        if m.ndim == 1:
            m = m[None, None, :]
        elif m.ndim == 2:
            m = m[:, None, :]
        x = x.masked_fill(~m, float("nan"))
    return x


def _as_covariates(val, bs, exp_len, name, device):
    if val is None:
        return None
    if isinstance(val, torch.Tensor):
        c = val.to(device=device, dtype=torch.float32)
    elif isinstance(val, np.ndarray):
        c = torch.as_tensor(val, dtype=torch.float32, device=device)
    else:
        arr = np.asarray(val)
        if arr.dtype.kind not in ("i", "u", "f", "b"):
            flat = arr.reshape(-1)
            uniq = {v: float(i) for i, v in enumerate(dict.fromkeys(flat.tolist()))}
            arr = np.vectorize(uniq.get, otypes=[np.float32])(arr)
        c = torch.as_tensor(arr, dtype=torch.float32, device=device)
    if c.ndim == 1:
        c = c[None, None, :]
    elif c.ndim == 2:
        c = c[None, :, :] if (bs == 1 and c.shape[1] == exp_len) else c[:, None, :]
    elif c.ndim != 3:
        raise ValueError(f"Expected 1D/2D/3D {name}, got {tuple(c.shape)}")
    if c.shape[0] == 1 and bs > 1:
        c = c.expand(bs, -1, -1)
    if c.shape[2] < exp_len:
        pad = torch.full((c.shape[0], c.shape[1], exp_len - c.shape[2]), float("nan"), device=device)
        c = torch.cat([pad, c], dim=2)
    elif c.shape[2] > exp_len:
        c = c[:, :, -exp_len:]
    return c


def _nonneg_mask(x):
    obs = torch.isfinite(x)
    has_any = obs.any(dim=-1)
    safe_min = torch.where(obs, x, torch.full_like(x, float("inf"))).amin(dim=-1)
    return has_any & (safe_min >= 0.0)


def _monotone_mask(x):
    """Detect strongly monotonic / cumulative series over the recent tail (last 32 obs)."""
    T = x.shape[-1]
    if T < 16:
        return torch.zeros(x.shape[:-1], dtype=torch.bool, device=x.device)
    tail = x[..., -min(T, 48):]
    d = tail[..., 1:] - tail[..., :-1]
    fin = torch.isfinite(d)
    cnt = fin.sum(dim=-1).float()
    pos_frac = ((d >= 0) & fin).sum(dim=-1).float() / cnt.clamp_min(1.0)
    neg_frac = ((d <= 0) & fin).sum(dim=-1).float() / cnt.clamp_min(1.0)
    nz = ((d.abs() > 1e-9) & fin).sum(dim=-1).float() >= 8.0
    return nz & (cnt >= 12.0) & ((pos_frac >= 0.94) | (neg_frac >= 0.94))


def _clean_1d_cpu(arr: np.ndarray, max_len: int = 8192) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    if len(a) > max_len:
        a = a[-max_len:]
    fin = np.flatnonzero(np.isfinite(a))
    if len(fin) == 0:
        return np.zeros(16, dtype=np.float32)
    a = a[fin[0] :].copy()
    bad = ~np.isfinite(a)
    if bad.any():
        good = np.flatnonzero(~bad)
        a[bad] = np.interp(np.flatnonzero(bad), good, a[good]).astype(np.float32)
    if len(a) < 16:
        a = np.pad(a, (16 - len(a), 0), mode="edge")
    return a.astype(np.float32, copy=False)


def _cutoff_mae_crps(q_pred, y_true, scale, q_levels):
    valid = torch.isfinite(y_true) & torch.isfinite(q_pred).all(dim=-1)
    n_v = valid.sum(dim=-1).float()
    has = (n_v > 0).float()
    denom = n_v.clamp_min(1.0)
    sc = scale.clamp_min(1e-6)
    y0 = torch.nan_to_num(y_true, nan=0.0)
    q0 = torch.nan_to_num(q_pred, nan=0.0)
    mae = ((y0 - q0[..., 4]).abs() * valid.float()).sum(dim=-1) / (denom * sc)
    diff = y0[..., None] - q0
    ql = q_levels.view(1, 1, -1)
    pb = 2.0 * torch.maximum(ql * diff, (ql - 1.0) * diff).mean(dim=-1)
    crps = (pb * valid.float()).sum(dim=-1) / (denom * sc)
    return torch.stack([mae, crps, has], dim=-1)


def mixture_cdf_quantiles(q_exp, w_exp):
    """Exact piecewise-linear Linear-Pool Mixture-CDF inversion for 9 deciles."""
    device, dtype = q_exp.device, q_exp.dtype
    N, H, M, Q = q_exp.shape
    p_inner = torch.linspace(0.1, 0.9, Q, device=device, dtype=dtype)
    q_lo = (q_exp[..., 0:1] - 0.5 * (q_exp[..., 1:2] - q_exp[..., 0:1])).clamp(max=q_exp[..., 0:1])
    q_hi = (q_exp[..., -1:] + 0.5 * (q_exp[..., -1:] - q_exp[..., -2:-1])).clamp(min=q_exp[..., -1:])
    knots = torch.cat([q_lo, q_exp, q_hi], dim=-1)
    p_knots = torch.cat([
        torch.zeros(1, device=device, dtype=dtype),
        p_inner,
        torch.ones(1, device=device, dtype=dtype),
    ])
    K = Q + 2

    k0 = knots[..., :-1]
    k1 = knots[..., 1:]
    span = (k1 - k0).clamp_min(1e-7)
    dp = (p_knots[1:] - p_knots[:-1]).view(1, 1, 1, K - 1, 1)

    lo = knots.amin(dim=(-2, -1))
    hi = knots.amax(dim=(-2, -1))
    lo_t = lo.unsqueeze(-1).expand(N, H, Q).clone()
    hi_t = hi.unsqueeze(-1).expand(N, H, Q).clone()
    target = p_inner.view(1, 1, Q)
    w_e = w_exp.unsqueeze(-1)

    for _ in range(18):
        mid = 0.5 * (lo_t + hi_t)
        frac = ((mid[:, :, None, None, :] - k0[..., None]) / span[..., None]).clamp(0.0, 1.0)
        cdf_m = (frac * dp).sum(dim=-2)
        cdf = (w_e * cdf_m).sum(dim=-2)
        below = cdf < target
        lo_t = torch.where(below, mid, lo_t)
        hi_t = torch.where(below, hi_t, mid)
    return 0.5 * (lo_t + hi_t)


@contextlib.contextmanager
def _skip_param_init():
    """Skip random weight initialization on Linear/Conv/Embedding/Norm layers before load_state_dict(assign=True).
    All non-persistent buffers (RoPE inv_freq, quantiles, patch windows) still initialize natively."""
    noop = lambda self, *a, **kw: None
    noop_fn = lambda t, *a, **kw: t
    patches = [
        (PreTrainedModel, "_init_weights", noop),
        (PreTrainedModel, "init_weights", noop),
        (nn.Linear, "reset_parameters", noop),
        (nn.Conv1d, "reset_parameters", noop),
        (nn.Embedding, "reset_parameters", noop),
        (nn.LayerNorm, "reset_parameters", noop),
        (nn.init, "kaiming_uniform_", noop_fn),
        (nn.init, "kaiming_normal_", noop_fn),
        (nn.init, "xavier_uniform_", noop_fn),
        (nn.init, "xavier_normal_", noop_fn),
        (nn.init, "normal_", noop_fn),
        (nn.init, "uniform_", noop_fn),
        (nn.init, "trunc_normal_", noop_fn),
    ]
    orig = [(obj, attr, getattr(obj, attr)) for obj, attr, _ in patches]
    try:
        for obj, attr, repl in patches:
            setattr(obj, attr, repl)
        yield
    finally:
        for obj, attr, val in orig:
            setattr(obj, attr, val)


def _fast_overlap_add(self, q_pred: torch.Tensor, original_length: int, stride: int) -> torch.Tensor:
    """Vectorized F.fold replacement for PatchTSTFM._overlap_add (eliminates per-patch Python loop)."""
    d_patch = self.config.d_patch
    B, _, Q = q_pred.shape
    n_patch = (original_length - d_patch) // stride + 1
    q_patches = q_pred.reshape(B, n_patch, d_patch, Q)
    win = self.patch_loss_window.to(device=q_pred.device, dtype=q_pred.dtype).view(1, 1, d_patch, 1)
    q_w = (q_patches * win).permute(0, 3, 2, 1).reshape(B * Q, d_patch, n_patch).contiguous()
    w_rep = win.view(1, d_patch, 1).expand(1, d_patch, n_patch).contiguous()
    orig_dtype = q_w.dtype
    accum = F.fold(q_w.float(), output_size=(1, original_length), kernel_size=(1, d_patch), stride=(1, stride))
    counts = F.fold(w_rep.float(), output_size=(1, original_length), kernel_size=(1, d_patch), stride=(1, stride))
    accum = accum.reshape(B, Q, original_length).transpose(1, 2)
    counts = counts.reshape(1, original_length, 1)
    return (accum / counts.clamp(min=1e-6)).to(orig_dtype)


def _fast_t5_ln_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(hidden_states, (hidden_states.shape[-1],), self.weight, self.variance_epsilon)


def _patch_iamtime_fast(iam_model: nn.Module) -> None:
    """Mathematically equivalent runtime speedups for iAmTime encoder:
    fused RMSNorm, SDPA (scale=1.0), precomputed RoPE/masks across layers, fused ex+q passes, zero H2D mask syncs."""
    if hasattr(F, "rms_norm"):
        for mod in iam_model.modules():
            if isinstance(mod, T5LayerNorm):
                mod.forward = types.MethodType(_fast_t5_ln_forward, mod)

    def _fast_t5_attn_rope_forward(
        self, hidden_states, mask, key_value_states=None, past_key_value=None,
        layer_head_mask=None, use_cache=False, output_attentions=False,
        cache_position=None, position_ids=None, cos_sin=None,
    ):
        batch_size = hidden_states.shape[0]
        is_cross = key_value_states is not None
        q = self.q(hidden_states).view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        cur = key_value_states if is_cross else hidden_states
        k = self.k(cur).view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        v = self.v(cur).view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        if self.use_rope:
            cos, sin = cos_sin if cos_sin is not None else self.rope_embed(v, position_ids)
            q, k = RoPE.apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
        if (layer_head_mask is None) and (not output_attentions) and (past_key_value is None):
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False, scale=1.0,
            )
        else:
            scores = torch.matmul(q, k.transpose(3, 2)) + mask
            w = F.softmax(scores.float(), dim=-1).type_as(scores)
            w = F.dropout(w, p=self.dropout, training=self.training)
            if layer_head_mask is not None:
                w = w * layer_head_mask
            out = torch.matmul(w, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
        return (self.o(out), past_key_value)

    def _fast_tok_read(self, Tok, Xflat, allow_mask, tok_valid_mask=None, pre_attn_mask=None, pre_tok_valid_f=None):
        N, T, d = Tok.shape
        K = Xflat.shape[1]
        Qh = self.q_proj(Tok).view(N, T, self.h, self.dk).transpose(1, 2)
        Kh = self.k_proj(Xflat).view(N, K, self.h, self.dk).transpose(1, 2)
        Vh = self.v_proj(Xflat).view(N, K, self.h, self.dk).transpose(1, 2)
        if pre_attn_mask is not None:
            attn_mask = pre_attn_mask
        elif allow_mask is None:
            attn_mask = None
        else:
            attn_mask = torch.zeros((N, 1, T, K), device=Tok.device, dtype=Tok.dtype).masked_fill_(
                ~allow_mask[:, None, :, :], torch.finfo(Tok.dtype).min
            )
        out = F.scaled_dot_product_attention(
            Qh, Kh, Vh, attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0, is_causal=False,
        )
        out = self.o_proj(out.transpose(1, 2).contiguous().view(N, T, d))
        if pre_tok_valid_f is not None:
            out = out * pre_tok_valid_f
        elif tok_valid_mask is not None:
            out = out * tok_valid_mask[:, :, None].to(out.dtype)
        return out

    def _fast_tok_self(self, Tok, mask, pre_attn_mask=None, pre_mask_f=None):
        N, T, d = Tok.shape
        Qh = self.q_proj(Tok).view(N, T, self.h, self.dk).transpose(1, 2)
        Kh = self.k_proj(Tok).view(N, T, self.h, self.dk).transpose(1, 2)
        Vh = self.v_proj(Tok).view(N, T, self.h, self.dk).transpose(1, 2)
        if pre_attn_mask is not None:
            attn_mask = pre_attn_mask
        else:
            attn_mask = torch.zeros((N, 1, 1, T), device=Tok.device, dtype=Tok.dtype).masked_fill_(
                ~mask[:, None, None, :], torch.finfo(Tok.dtype).min
            )
        out = F.scaled_dot_product_attention(
            Qh, Kh, Vh, attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0, is_causal=False,
        )
        out = self.o_proj(out.transpose(1, 2).contiguous().view(N, T, d))
        return out * (pre_mask_f if pre_mask_f is not None else mask[:, :, None].to(out.dtype))

    def _fast_cross_ex(self, q_tokens, ex_tokens):
        B, E, two, d = ex_tokens.shape
        return self.attn(Tok=q_tokens, Xflat=ex_tokens.reshape(B, E * two, d), allow_mask=None, tok_valid_mask=None)

    for blk in iam_model.encoder.block:
        blk.time_series_attn.SelfAttention.forward = types.MethodType(
            _fast_t5_attn_rope_forward, blk.time_series_attn.SelfAttention
        )
        blk.example_fusion_attn.SelfAttention.forward = types.MethodType(
            _fast_t5_attn_rope_forward, blk.example_fusion_attn.SelfAttention
        )
        blk.token_read_cross_attn.forward = types.MethodType(_fast_tok_read, blk.token_read_cross_attn)
        blk.token_self_attn.forward = types.MethodType(_fast_tok_self, blk.token_self_attn)
        blk.example_fusion_cross_attn.attn.forward = types.MethodType(_fast_tok_read, blk.example_fusion_cross_attn.attn)
        blk.example_fusion_cross_attn.forward = types.MethodType(_fast_cross_ex, blk.example_fusion_cross_attn)

    orig_stack_forward = iam_model.encoder.forward

    def _fast_stack_forward(
        self, X_ex, M_ex, Tok_ex, Mtok_ex, allow_ex,
        X_q, M_q, Tok_q, Mtok_q, allow_q,
        start_idx: int, mid_idx: int, P_h_q: int, output_attentions: bool = False,
    ):
        B, E, Sx, P, d = X_ex.shape
        _, _, Sq, Pq, _ = X_q.shape
        if output_attentions or self.training or E == 0 or P != Pq:
            return orig_stack_forward(
                X_ex, M_ex, Tok_ex, Mtok_ex, allow_ex,
                X_q, M_q, Tok_q, Mtok_q, allow_q,
                start_idx=start_idx, mid_idx=mid_idx, P_h_q=P_h_q, output_attentions=output_attentions,
            )
        X_ex = self.dropout(X_ex)
        X_q = self.dropout(X_q)
        T = Tok_ex.shape[2]
        dtype, dev = X_q.dtype, X_q.device
        fmin = torch.finfo(dtype).min

        def _add_mask_2d(m2d):
            return torch.zeros((m2d.shape[0], 1, 1, m2d.shape[1]), device=dev, dtype=dtype).masked_fill_(
                ~m2d[:, None, None, :], fmin
            )

        Mp_both = torch.cat([M_ex.reshape(B * E * Sx, P).bool(), M_q.reshape(B * Sq, Pq).bool()], dim=0)
        mask_ts_both = _add_mask_2d(Mp_both)
        pos_ids = torch.arange(0, P, dtype=torch.long, device=dev).unsqueeze(0)
        cos_sin = self.block[0].time_series_attn.SelfAttention.rope_embed(X_q, pos_ids)

        Ms = torch.cat([
            M_ex.permute(0, 3, 1, 2).reshape(B * P, E * Sx).bool(),
            M_q.squeeze(1).permute(0, 2, 1).reshape(B * P, Sq).bool(),
        ], dim=1)
        mask_ef = _add_mask_2d(Ms)

        same_s = (Sx == Sq)
        Mtok_ex_b = Mtok_ex.reshape(B * E, T).bool()
        Mtok_q_b = Mtok_q[:, 0, :].bool()
        Mtok_both_b = torch.cat([Mtok_ex_b, Mtok_q_b], dim=0)
        mask_tself_both = _add_mask_2d(Mtok_both_b)
        mtok_both_f = Mtok_both_b[:, :, None].to(dtype)

        if same_s:
            allow_both = torch.cat([allow_ex, allow_q], dim=0)
            mask_tread_both = torch.zeros((allow_both.shape[0], 1, T, Sx * P), device=dev, dtype=dtype).masked_fill_(
                ~allow_both[:, None, :, :], fmin
            )
        else:
            mask_tread_ex = torch.zeros((B * E, 1, T, Sx * P), device=dev, dtype=dtype).masked_fill_(
                ~allow_ex[:, None, :, :], fmin
            )
            mask_tread_q = torch.zeros((B, 1, T, Sq * P), device=dev, dtype=dtype).masked_fill_(
                ~allow_q[:, None, :, :], fmin
            )
            mtok_ex_f = Mtok_ex_b[:, :, None].to(dtype)
            mtok_q_f = Mtok_q_b[:, :, None].to(dtype)

        n_ex_s, n_q_s, n_ex_b = B * E * Sx, B * Sq, B * E

        for blk in self.block:
            X_both = torch.cat([X_ex.reshape(n_ex_s, P, d), X_q.reshape(n_q_s, P, d)], dim=0)
            X_both = X_both + blk.time_series_attn.dropout(
                blk.time_series_attn.SelfAttention(
                    blk.time_series_attn.layer_norm(X_both), mask=mask_ts_both, cos_sin=cos_sin
                )[0]
            )
            X_ex = X_both[:n_ex_s].reshape(B, E, Sx, P, d)
            X_q = X_both[n_ex_s:].reshape(B, 1, Sq, P, d)

            X_ef = torch.cat([
                X_ex.permute(0, 3, 1, 2, 4).reshape(B * P, E * Sx, d),
                X_q.squeeze(1).permute(0, 2, 1, 3).reshape(B * P, Sq, d),
            ], dim=1)
            X_ef = X_ef + blk.example_fusion_attn.dropout(
                blk.example_fusion_attn.SelfAttention(blk.example_fusion_attn.layer_norm(X_ef), mask=mask_ef)[0]
            )
            X_ex_ef, X_q_ef = X_ef.split([E * Sx, Sq], dim=1)
            X_ex = X_ex_ef.reshape(B, P, E, Sx, d).permute(0, 2, 3, 1, 4).contiguous()
            X_q = X_q_ef.reshape(B, P, 1, Sq, d).permute(0, 2, 3, 1, 4).contiguous()

            Tok_both = torch.cat([Tok_ex.reshape(n_ex_b, T, d), Tok_q[:, 0, :, :]], dim=0)
            if same_s:
                X_flat_both = torch.cat([X_ex.reshape(n_ex_b, Sx * P, d), X_q.reshape(B, Sq * P, d)], dim=0)
                Tok_both = Tok_both + blk.token_read_cross_attn(
                    Tok=blk.ln(Tok_both), Xflat=blk.ln(X_flat_both),
                    allow_mask=None, pre_attn_mask=mask_tread_both, pre_tok_valid_f=mtok_both_f,
                )
            else:
                te = Tok_both[:n_ex_b] + blk.token_read_cross_attn(
                    Tok=blk.ln(Tok_both[:n_ex_b]), Xflat=blk.ln(X_ex.reshape(n_ex_b, Sx * P, d)),
                    allow_mask=None, pre_attn_mask=mask_tread_ex, pre_tok_valid_f=mtok_ex_f,
                )
                tq = Tok_both[n_ex_b:] + blk.token_read_cross_attn(
                    Tok=blk.ln(Tok_both[n_ex_b:]), Xflat=blk.ln(X_q.reshape(B, Sq * P, d)),
                    allow_mask=None, pre_attn_mask=mask_tread_q, pre_tok_valid_f=mtok_q_f,
                )
                Tok_both = torch.cat([te, tq], dim=0)

            Tok_both = Tok_both + blk.token_self_attn(
                blk.ln(Tok_both), mask=None, pre_attn_mask=mask_tself_both, pre_mask_f=mtok_both_f,
            )
            Tok_ex = Tok_both[:n_ex_b].reshape(B, E, T, d)
            Tok_q_blk = Tok_both[n_ex_b:]

            q_sm = torch.stack([Tok_q_blk[:, start_idx, :], Tok_q_blk[:, mid_idx, :]], dim=1)
            ex_sm = torch.stack([Tok_ex[:, :, start_idx, :], Tok_ex[:, :, mid_idx, :]], dim=2)
            q_sm = q_sm + blk.example_fusion_cross_attn(q_tokens=blk.ln(q_sm), ex_tokens=blk.ln(ex_sm))
            Tok_q_blk[:, start_idx, :] = q_sm[:, 0, :]
            Tok_q_blk[:, mid_idx, :] = q_sm[:, 1, :]
            Tok_q = Tok_q_blk.unsqueeze(1)

            X_q = blk.token_write(X_q=X_q, start_token=q_sm[:, 0, :], mid_token=q_sm[:, 1, :], num_hist_patches=P_h_q)

            X_p_both = blk.ffn_patches(torch.cat([X_ex.reshape(n_ex_s, P, d), X_q.reshape(n_q_s, P, d)], dim=0))
            X_ex = X_p_both[:n_ex_s].reshape(B, E, Sx, P, d)
            X_q = X_p_both[n_ex_s:].reshape(B, 1, Sq, P, d)

            Tok_f_both = blk.ffn_tokens(torch.cat([Tok_ex.reshape(n_ex_b, T, d), Tok_q_blk], dim=0))
            Tok_ex = Tok_f_both[:n_ex_b].reshape(B, E, T, d)
            Tok_q = Tok_f_both[n_ex_b:].unsqueeze(1)

        return self.dropout(self.ln(X_ex)), Tok_ex, self.dropout(self.ln(X_q)), Tok_q

    iam_model.encoder.forward = types.MethodType(_fast_stack_forward, iam_model.encoder)


class ZitraV3Forecaster:
    """Self-contained 4-architecture Dynamic-32K Pure Apache-2.0 probabilistic forecaster."""

    def __init__(self, t0_model, chronos_pipe, patchtst_model, iamtime_pipe, policy=None):
        self.t0 = t0_model
        self.chronos = chronos_pipe
        self.patchtst = patchtst_model
        self.iamtime = iamtime_pipe
        self.device = next(t0_model.parameters()).device
        self.policy = policy or {
            "version": "zitra-titan-v3",
            "license": "Apache-2.0",
            "t0_dynamic_scale_base": True,
            "t0_max_context_known": 16384,
            "t0_max_context_default": 8192,
            "prior_long": [0.42, 0.16, 0.30, 0.12],
            "prior_short": [0.58, 0.24, 0.02, 0.16],
            "prior_mv_long": [0.50, 0.20, 0.10, 0.20],
            "prior_mv_short": [0.54, 0.24, 0.02, 0.20],
            "prior_cov_long": [0.48, 0.20, 0.00, 0.32],
            "prior_cov_short": [0.48, 0.24, 0.00, 0.28],
            "short_lo": 80,
            "short_hi": 160,
            "skip_pt_below": 64,
            "skip_pt_on_known_cov": True,
            "skip_pt_on_cov": False,
            "skip_pt_on_multivar": False,
            "iam_mv_as_cov": True,
            "iam_mv_max_v": 24,
            "iam_monotone_damp": 0.15,
            "w_cap_xlong": 192,
            "w_cap_long": 96,
            "w_cap_short": 48,
            "fast_bt": True,
            "t0_bf16_amp": True,
            "parallel_streams": False,
            "ref_mode": "median",
            "cross_learning": "auto",
            "ci_max_t": 1024,
            "ci_max_group": 64,
            "ci_bt_margin": -0.012,
            "tau": 11.0,
            "kappa": 16.0,
            "disagree_pen": 0.40,
            "min_bt_w": 8,
            "replay": True,
            "mixture_cdf": True,
            "integer_snap": True,
            "nonneg_clip": True,
        }
        self.max_context = int(max(
            self.policy.get("t0_max_context_known", 24576),
            self.policy.get("t0_max_context_default", 8192),
        ))
        self.quantile_levels = QUANTILES_21
        self._q_t = torch.tensor(DECILES, dtype=torch.float32)
        self.last_routing_weights: np.ndarray | None = None
        self.last_expert_medians: dict[str, np.ndarray] | None = None
        self.last_cross_learning_used: dict[str, bool] = {"t0": False, "chronos2": False}
        self.last_t0_ctx_16k_frac: float = 0.0
        self._is_cuda = torch.cuda.is_available() and str(self.device).startswith("cuda")
        self._high_vram = False
        self._streams: list[torch.cuda.Stream] | None = None
        if self._is_cuda:
            try:
                props = torch.cuda.get_device_properties(self.device)
                self._high_vram = (props.total_memory >= 40 * (1024 ** 3))
                self._streams = [torch.cuda.Stream(device=self.device) for _ in range(4)]
            except Exception:
                self._high_vram = False
                self._streams = None

    @classmethod
    def from_pretrained(cls, path, device="cuda"):
        pkg = Path(path)
        policy = json.loads((pkg / "policy.json").read_text())
        is_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
        dev_str = str(torch.device(device))
        if is_cuda:
            torch.set_float32_matmul_precision("high")

        with safe_open(str(pkg / "model.safetensors"), framework="pt", device=dev_str) as f:
            keys = list(f.keys())
            with _skip_param_init():
                # 0. Zitra-t0-256M-Dyn32K (fp32 weights: TF32 on T<=2048 for speed & precision, bf16 amp on T>2048)
                t0_cfg = json.loads((pkg / "t0_config.json").read_text())
                t0_sd = {k[3:]: f.get_tensor(k).float() for k in keys if k.startswith("t0.")}
                t0_model = T0Forecaster(**t0_cfg)
                t0_model.load_state_dict(t0_sd, strict=True, assign=True)
                del t0_sd
                t0_model.transformer.rotary_emb.scale_base = 512.0
                t0_model = t0_model.to(device=device).eval()

                # 1. Zitra-Chronos2-120M (native bf16 on CUDA, fp32 on CPU + fused RMSNorm)
                ch_cfg = Chronos2CoreConfig(**json.loads((pkg / "chronos_config.json").read_text()))
                ch_sd = {
                    k[8:]: (f.get_tensor(k) if is_cuda else f.get_tensor(k).float())
                    for k in keys if k.startswith("chronos.")
                }
                ch_model = Chronos2Model(ch_cfg)
                ch_model.load_state_dict(ch_sd, strict=True, assign=True)
                del ch_sd
                ch_model = ch_model.to(device=device).eval()
                if hasattr(F, "rms_norm"):
                    for mod in ch_model.modules():
                        if isinstance(mod, T5LayerNorm):
                            mod.forward = types.MethodType(_fast_t5_ln_forward, mod)

                # 2. Zitra-PatchTST-385M (fp32 weights + bf16 autocast preserves RevIN/LayerNorm precision; vectorized F.fold _overlap_add)
                pt_cfg = PatchTSTFMConfig(**json.loads((pkg / "patchtst_config.json").read_text()))
                pt_sd = {k[9:]: f.get_tensor(k).float() for k in keys if k.startswith("patchtst.")}
                pt_model = PatchTSTFMForPrediction(pt_cfg)
                pt_model.load_state_dict(pt_sd, strict=True, assign=True)
                del pt_sd
                pt_model = pt_model.to(device=device).eval()
                if hasattr(pt_model, "backbone") and hasattr(pt_model.backbone, "_overlap_add"):
                    pt_model.backbone._overlap_add = types.MethodType(_fast_overlap_add, pt_model.backbone)

                # 3. Zitra-iAmTime-302M (native bf16 on CUDA + fused encoder stack & explicit RoPE/quantiles restore)
                iam_cfg = T5Config.from_dict(json.loads((pkg / "iamtime_config.json").read_text()))
                iam_sd = {
                    k[8:]: (f.get_tensor(k) if is_cuda else f.get_tensor(k).float())
                    for k in keys if k.startswith("iamtime.")
                }
                iam_model = iAmTimeModel(iam_cfg)
                iam_model.load_state_dict(iam_sd, strict=True, assign=True)
                del iam_sd
                iam_dtype = torch.bfloat16 if is_cuda else torch.float32
                iam_model = iam_model.to(device=device).eval()
                for mod in iam_model.modules():
                    if isinstance(mod, RoPE):
                        inv_freq = 1.0 / (
                            mod.base ** (torch.arange(0, mod.dim, 2, dtype=torch.int64, device=device).float() / mod.dim)
                        )
                        mod.register_buffer("inv_freq", inv_freq, persistent=False)
                iam_model.register_buffer(
                    "quantiles",
                    torch.tensor(iam_model.iamtime_config.quantiles, device=device, dtype=iam_dtype),
                    persistent=False,
                )
                _patch_iamtime_fast(iam_model)

        return cls(t0_model, Chronos2Pipeline(ch_model), pt_model, iAmTimePipeline(iam_model), policy=policy)

    def _set_t0_scale_base(self, t_sl: int):
        if not self.policy.get("t0_dynamic_scale_base", True):
            self.t0.transformer.rotary_emb.scale_base = float(self.policy.get("t0_scale_base", 512.0))
            return
        if t_sl <= 8192:
            self.t0.transformer.rotary_emb.scale_base = 512.0
        elif t_sl <= 16384:
            self.t0.transformer.rotary_emb.scale_base = 1536.0
        else:
            self.t0.transformer.rotary_emb.scale_base = 2048.0

    def _t0_use_amp(self, x_tensor: torch.Tensor, t_sl: int) -> bool:
        if not (self._is_cuda and bool(self.policy.get("t0_bf16_amp", True))):
            return False
        bf16_min_t = int(self.policy.get("bf16_min_t", 2048))
        if t_sl <= bf16_min_t:
            return False
        if bool(self.policy.get("t0_high_offset_fp32", True)):
            fin = torch.isfinite(x_tensor)
            if fin.any():
                xf = x_tensor.float().masked_fill(~fin, 0.0)
                cnt = fin.sum(dim=-1).float().clamp_min(1.0)
                mu = xf.sum(dim=-1) / cnt
                abs_mu = xf.abs().sum(dim=-1) / cnt
                std = ((xf - mu[..., None]).square() * fin.float()).sum(dim=-1).div(cnt).sqrt().clamp_min(1e-6)
                if float((abs_mu / std).max().item()) > 25.0:
                    return False
        return True

    @torch.inference_mode()
    def _run_t0(self, x, h, past_cov, known_cov, max_ctx=8192, return_21=False, sym=True):
        B, V, T = x.shape
        device = x.device
        t_sl = min(T, int(max_ctx))
        self._set_t0_scale_base(t_sl)
        t0_start = T - t_sl
        x8 = x[..., t0_start:]
        p8 = past_cov[..., t0_start:] if past_cov is not None else None
        k8 = known_cov[..., t0_start : T + h] if known_cov is not None else None
        q_req = QUANTILES_21 if return_21 else DECILES
        n_q = len(q_req)
        has_cov = (p8 is not None) or (k8 is not None)
        use_amp = self._t0_use_amp(x8, t_sl)

        def _exec_t0(x_sub, p_sub, k_sub):
            v_sub = x_sub.shape[1]
            max_b = max(1, (8 if t_sl > 16384 else (16 if t_sl > 8192 else 64)) // max(1, v_sub))
            parts_b = []
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                for b0 in range(0, B, max_b):
                    xb = x_sub[b0 : b0 + max_b]
                    pb = p_sub[b0 : b0 + max_b] if p_sub is not None else None
                    kb = k_sub[b0 : b0 + max_b] if k_sub is not None else None
                    bc = xb.shape[0]
                    if not has_cov:
                        blocks = []
                        for i in range(bc):
                            row = xb[i]
                            ii = torch.where(row.isfinite().any(0))[0]
                            s = int(ii[0]) if len(ii) else t_sl - 1
                            blocks.append(row[:, s:])
                        if sym and (bc * v_sub <= 32) and (t_sl <= 8192):
                            arr, mask, ids = batch_series(blocks + [-b for b in blocks])
                            out_all = self.t0.predict(
                                arr.to(device), horizon=h, mask=mask.to(device), group_ids=ids.to(device),
                                quantile_levels=q_req, trim_leading_missing=True, symmetric_averaging=False,
                                clip_nonnegative=False, snap_integers=False,
                            ).quantiles.float().reshape(2, bc, v_sub, h, n_q)
                            out = 0.5 * (out_all[0] + (-out_all[1]).flip(-1))
                            nn_b = _nonneg_mask(xb)
                            out = torch.sort(torch.where(nn_b[..., None, None], out.clamp_min(0.0), out), dim=-1).values
                            parts_b.append(out)
                        else:
                            arr, mask, ids = batch_series(blocks)
                            out = self.t0.predict(
                                arr.to(device), horizon=h, mask=mask.to(device), group_ids=ids.to(device),
                                quantile_levels=q_req, trim_leading_missing=True, symmetric_averaging=sym,
                                clip_nonnegative=True, snap_integers=True,
                            ).quantiles.float()
                            parts_b.append(out.reshape(bc, v_sub, h, n_q))
                    else:
                        blocks = []
                        for i in range(bc):
                            row = xb[i]
                            valid = row.isfinite().any(0)
                            if pb is not None:
                                valid = valid | pb[i].isfinite().any(0)
                            if kb is not None:
                                valid = valid | kb[i, :, :t_sl].isfinite().any(0)
                            ii = torch.where(valid)[0]
                            s = int(ii[0]) if len(ii) else t_sl - 1
                            cl = t_sl - s
                            fc_i = kb[i : i + 1, :, s:].cpu() if kb is not None else None
                            ts = TimeSeries.from_array(row[None, :, s:].cpu(), future_covariates=fc_i, horizon=h)
                            if pb is not None:
                                c_p = pb.shape[1]
                                w_len = ts.seq_len
                                p_vals = pb[i, :, s:].cpu()
                                hist = torch.zeros((c_p, w_len), dtype=torch.float32)
                                hist[:, :cl] = torch.nan_to_num(p_vals, nan=0.0)
                                pmask = torch.full((c_p, w_len), MaskType.MISSING, dtype=ts.mask.dtype)
                                pmask[:, :cl] = torch.where(
                                    torch.isnan(p_vals),
                                    torch.tensor(MaskType.MISSING, dtype=ts.mask.dtype),
                                    torch.tensor(MaskType.VALID, dtype=ts.mask.dtype),
                                )
                                ts = TimeSeries(
                                    variates=torch.cat([ts.variates, hist], dim=0),
                                    mask=torch.cat([ts.mask, pmask], dim=0),
                                    group_ids=torch.cat([ts.group_ids, torch.zeros_like(pmask, dtype=ts.group_ids.dtype)], dim=0),
                                    variate_type=torch.cat(
                                        [ts.variate_type, torch.full_like(pmask, VariateType.HISTORICAL, dtype=ts.variate_type.dtype)], dim=0
                                    ),
                                )
                            blocks.append(ts)
                        batch = TimeSeries.batch(blocks).to(device)
                        out = self.t0.predict(
                            batch, horizon=h, context_length=batch.seq_len - h, quantile_levels=q_req,
                            trim_leading_missing=True, symmetric_averaging=sym,
                            clip_nonnegative=True, snap_integers=True,
                        ).quantiles.float()
                        parts_b.append(out.reshape(bc, v_sub, h, n_q))
            return torch.cat(parts_b, dim=0)

        total_rows = V + (p8.shape[1] if p8 is not None else 0)
        if total_rows <= 32 or V <= 24:
            return _exec_t0(x8, p8, k8)
        parts = []
        for v0 in range(0, V, 24):
            v1 = min(V, v0 + 24)
            parts.append(_exec_t0(x8[:, v0:v1], p8, k8))
        return torch.cat(parts, dim=1)

    @torch.inference_mode()
    def _run_t0_ci(self, x, h, return_21=False, max_grp=64, sym=True):
        B, V, T = x.shape
        device = x.device
        self._set_t0_scale_base(min(T, 8192))
        q_req = QUANTILES_21 if return_21 else DECILES
        n_q = len(q_req)
        use_amp = self._t0_use_amp(x, T)
        parts = []
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            for i0 in range(0, B, max_grp):
                sub = x[i0 : i0 + max_grp, 0, :]
                ii = torch.where(sub.isfinite().any(0))[0]
                s = int(ii[0]) if len(ii) else T - 1
                arr, mask, ids = batch_series([sub[:, s:]])
                out = self.t0.predict(
                    arr.to(device), horizon=h, mask=mask.to(device), group_ids=ids.to(device),
                    quantile_levels=q_req, trim_leading_missing=True, symmetric_averaging=sym,
                    clip_nonnegative=True, snap_integers=True,
                ).quantiles.float()
                parts.append(out.reshape(sub.shape[0], 1, h, n_q))
        return torch.cat(parts, dim=0)

    @torch.inference_mode()
    def _run_t0_fused_bt_nocov(self, bt_slices, w_bt, max_ctx=8192, sym=False):
        """Fuse equal-horizon backtest cuts into a single fast t0 forward pass (sym=False by default!)."""
        K = len(bt_slices)
        B, V = bt_slices[0].shape[:2]
        device = bt_slices[0].device
        if V > 32 or max_ctx > 8192:
            return [self._run_t0(sx, w_bt, None, None, max_ctx=max_ctx, return_21=False, sym=sym) for sx in bt_slices]
        self._set_t0_scale_base(min(bt_slices[0].shape[2], max_ctx))
        use_amp = self._t0_use_amp(bt_slices[0], min(bt_slices[0].shape[2], max_ctx))
        blocks = []
        for sx in bt_slices:
            t_len = sx.shape[2]
            t_sl = min(t_len, max_ctx)
            x8 = sx[..., t_len - t_sl :]
            for i in range(B):
                row = x8[i]
                ii = torch.where(row.isfinite().any(0))[0]
                s = int(ii[0]) if len(ii) else t_sl - 1
                blocks.append(row[:, s:])
        arr, mask, ids = batch_series(blocks)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            out = self.t0.predict(
                arr.to(device), horizon=w_bt, mask=mask.to(device), group_ids=ids.to(device),
                quantile_levels=DECILES, trim_leading_missing=True, symmetric_averaging=sym,
                clip_nonnegative=True, snap_integers=True,
            ).quantiles.float().reshape(K, B, V, w_bt, 9)
        return [out[k] for k in range(K)]

    @torch.inference_mode()
    def _run_t0_fused_bt_cov(self, bt_slices, w_bt, bt_pcs, bt_kcs, max_ctx=8192, sym=False):
        """Fuse equal-horizon covariate backtest cuts [bt0, bt1] into a single TimeSeries.batch pass."""
        K = len(bt_slices)
        B, V = bt_slices[0].shape[:2]
        device = bt_slices[0].device
        total_rows = V + (bt_pcs[0].shape[1] if bt_pcs[0] is not None else 0)
        t_sl_max = max(min(sx.shape[2], int(max_ctx)) for sx in bt_slices)
        max_b = max(1, (8 if t_sl_max > 16384 else (16 if t_sl_max > 8192 else 64)) // max(1, V))
        if K < 2 or (total_rows > 32 and V > 24) or (K * B > max_b):
            return [
                self._run_t0(bt_slices[ci], w_bt, bt_pcs[ci], bt_kcs[ci], max_ctx=max_ctx, return_21=False, sym=sym)
                for ci in range(K)
            ]
        self._set_t0_scale_base(t_sl_max)
        use_amp = self._t0_use_amp(bt_slices[-1][..., max(0, bt_slices[-1].shape[2] - t_sl_max):], t_sl_max)
        blocks = []
        for ci in range(K):
            sx, pc, kc = bt_slices[ci], bt_pcs[ci], bt_kcs[ci]
            t_len = sx.shape[2]
            t_sl = min(t_len, int(max_ctx))
            t0_start = t_len - t_sl
            x8 = sx[..., t0_start:]
            p8 = pc[..., t0_start:] if pc is not None else None
            k8 = kc[..., t0_start : t_len + w_bt] if kc is not None else None
            for i in range(B):
                row = x8[i]
                valid = row.isfinite().any(0)
                if p8 is not None:
                    valid = valid | p8[i].isfinite().any(0)
                if k8 is not None:
                    valid = valid | k8[i, :, :t_sl].isfinite().any(0)
                ii = torch.where(valid)[0]
                s = int(ii[0]) if len(ii) else t_sl - 1
                cl = t_sl - s
                fc_i = k8[i : i + 1, :, s:].cpu() if k8 is not None else None
                ts = TimeSeries.from_array(row[None, :, s:].cpu(), future_covariates=fc_i, horizon=w_bt)
                if p8 is not None:
                    c_p = p8.shape[1]
                    w_len = ts.seq_len
                    p_vals = p8[i, :, s:].cpu()
                    hist = torch.zeros((c_p, w_len), dtype=torch.float32)
                    hist[:, :cl] = torch.nan_to_num(p_vals, nan=0.0)
                    pmask = torch.full((c_p, w_len), MaskType.MISSING, dtype=ts.mask.dtype)
                    pmask[:, :cl] = torch.where(
                        torch.isnan(p_vals),
                        torch.tensor(MaskType.MISSING, dtype=ts.mask.dtype),
                        torch.tensor(MaskType.VALID, dtype=ts.mask.dtype),
                    )
                    ts = TimeSeries(
                        variates=torch.cat([ts.variates, hist], dim=0),
                        mask=torch.cat([ts.mask, pmask], dim=0),
                        group_ids=torch.cat([ts.group_ids, torch.zeros_like(pmask, dtype=ts.group_ids.dtype)], dim=0),
                        variate_type=torch.cat(
                            [ts.variate_type, torch.full_like(pmask, VariateType.HISTORICAL, dtype=ts.variate_type.dtype)], dim=0
                        ),
                    )
                blocks.append(ts)
        batch = TimeSeries.batch(blocks).to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            out = self.t0.predict(
                batch, horizon=w_bt, context_length=batch.seq_len - w_bt, quantile_levels=DECILES,
                trim_leading_missing=True, symmetric_averaging=sym,
                clip_nonnegative=True, snap_integers=True,
            ).quantiles.float().reshape(K, B, V, w_bt, 9)
        return [out[k] for k in range(K)]

    @torch.inference_mode()
    def _run_chronos(self, x, h, past_cov, known_cov, fallback=None, sym=True):
        B, V, T = x.shape
        device = x.device
        t_sl = min(T, 8192)
        t0_start = T - t_sl
        x8 = x[..., t0_start:].detach().cpu().float()
        p8 = past_cov[..., t0_start:].detach().cpu().float() if past_cov is not None else None
        k8 = known_cov[..., t0_start : T + h].detach().cpu().float() if known_cov is not None else None
        has_cov = (p8 is not None) or (k8 is not None)

        try:
            if not has_cov:
                rows = []
                for i in range(B):
                    row = x8[i]
                    ii = torch.where(row.isfinite().any(0))[0]
                    s = int(ii[0]) if len(ii) else t_sl - 1
                    rows.append(row[:, s:])
                if sym:
                    p12, _ = self.chronos.predict_quantiles(
                        rows + [-r for r in rows], prediction_length=h, quantile_levels=DECILES, cross_learning=False
                    )
                    q = 0.5 * (torch.stack(p12[:B]).float() + (-torch.stack(p12[B:]).float()).flip(-1))
                else:
                    p1, _ = self.chronos.predict_quantiles(
                        rows, prediction_length=h, quantile_levels=DECILES, cross_learning=False
                    )
                    q = torch.stack(p1).float()
            else:
                inps = []
                for i in range(B):
                    row = x8[i]
                    ii = torch.where(row.isfinite().any(0))[0]
                    s = int(ii[0]) if len(ii) else t_sl - 1
                    d = {"target": row[:, s:].numpy()}
                    pc, fc = {}, {}
                    if p8 is not None:
                        for c in range(p8.shape[1]):
                            arr = p8[i, c, s:].numpy()
                            if np.isfinite(arr).any(): pc[f"p{c}"] = arr
                    if k8 is not None:
                        for c in range(k8.shape[1]):
                            arr_p = k8[i, c, s:t_sl].numpy()
                            arr_f = k8[i, c, t_sl : t_sl + h].numpy()
                            if np.isfinite(arr_p).any() and np.isfinite(arr_f).any():
                                pc[f"k{c}"] = arr_p; fc[f"k{c}"] = arr_f
                    if pc: d["past_covariates"] = pc
                    if fc: d["future_covariates"] = fc
                    inps.append(d)
                p1, _ = self.chronos.predict_quantiles(inps, prediction_length=h, quantile_levels=DECILES, cross_learning=False)
                q = torch.stack(p1).float()
            q = q.to(device)
            all_nonneg = _nonneg_mask(x)
            q = torch.where(all_nonneg[..., None, None], q.clamp_min(0.0), q)
            q = torch.sort(q, dim=-1).values
            if fallback is not None:
                bad = ~torch.isfinite(q).all(dim=(-2, -1))
                if bad.any():
                    q = torch.where(bad[..., None, None], fallback, q)
            return q
        except Exception:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return fallback.clone() if fallback is not None else None

    @torch.inference_mode()
    def _run_chronos_ci(self, x, h, fallback, max_grp=64, sym=True):
        B, V, T = x.shape
        device = x.device
        t_sl = min(T, 8192)
        x8 = x[..., T - t_sl :].detach().cpu().float()
        try:
            out_parts = []
            for i0 in range(0, B, max_grp):
                sub = x8[i0 : i0 + max_grp]
                bc = sub.shape[0]
                rows_p = []
                for i in range(bc):
                    row = sub[i]
                    ii = torch.where(row.isfinite().any(0))[0]
                    s = int(ii[0]) if len(ii) else t_sl - 1
                    rows_p.append(row[:, s:])
                p1, _ = self.chronos.predict_quantiles(rows_p, prediction_length=h, quantile_levels=DECILES, cross_learning=True)
                if sym:
                    p2, _ = self.chronos.predict_quantiles([-r for r in rows_p], prediction_length=h, quantile_levels=DECILES, cross_learning=True)
                    q = 0.5 * (torch.stack(p1).float() + (-torch.stack(p2).float()).flip(-1))
                else:
                    q = torch.stack(p1).float()
                out_parts.append(q)
            q = torch.cat(out_parts, dim=0).to(device)
            all_nonneg = _nonneg_mask(x)
            q = torch.where(all_nonneg[..., None, None], q.clamp_min(0.0), q)
            q = torch.sort(q, dim=-1).values
            bad = ~torch.isfinite(q).all(dim=(-2, -1))
            if bad.any():
                q = torch.where(bad[..., None, None], fallback, q)
            return q
        except Exception:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return fallback.clone()

    @torch.inference_mode()
    def _run_chronos_fused_bt_nocov(self, slices_x, w_bt, sym=False):
        K = len(slices_x)
        B, V = slices_x[0].shape[:2]
        device = slices_x[0].device
        try:
            rows_pos = []
            for sx in slices_x:
                t_len = sx.shape[2]
                t_sl = min(t_len, 8192)
                x8 = sx[..., t_len - t_sl :].detach().cpu().float()
                for i in range(B):
                    row = x8[i]
                    ii = torch.where(row.isfinite().any(0))[0]
                    s = int(ii[0]) if len(ii) else t_sl - 1
                    rows_pos.append(row[:, s:])
            KB = K * B
            if sym:
                p12, _ = self.chronos.predict_quantiles(
                    rows_pos + [-r for r in rows_pos], prediction_length=w_bt, quantile_levels=DECILES, cross_learning=False
                )
                q_all = 0.5 * (torch.stack(p12[:KB]).float() + (-torch.stack(p12[KB:]).float()).flip(-1))
            else:
                p1, _ = self.chronos.predict_quantiles(
                    rows_pos, prediction_length=w_bt, quantile_levels=DECILES, cross_learning=False
                )
                q_all = torch.stack(p1).float()
            q_all = q_all.reshape(K, B, V, w_bt, 9).to(device)
            res = []
            for k in range(K):
                qk = q_all[k]
                all_nonneg = _nonneg_mask(slices_x[k])
                qk = torch.where(all_nonneg[..., None, None], qk.clamp_min(0.0), qk)
                res.append(torch.sort(qk, dim=-1).values)
            return res
        except Exception:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return [None] * K

    @torch.inference_mode()
    def _run_patchtst_slices(self, slices_x, hs, syms=None):
        """PatchTST-FM-r2 with exact past_values API & fused single-pass [main, -main, bt0, bt1] execution!"""
        K = len(slices_x)
        if syms is None:
            syms = [True] * K
        B, V = slices_x[0].shape[:2]
        N = B * V
        device = slices_x[0].device
        pt_dtype = next(self.patchtst.parameters()).dtype
        use_amp = self._is_cuda and (pt_dtype == torch.float32)
        ch_bs_base = 64 if self._high_vram else 16
        ch_bs_big = 32 if self._high_vram else 8

        def _extract_q(qo, hk):
            if isinstance(qo, list):
                ps = []
                for t in qo:
                    a = t.float()
                    while a.ndim > 2 and a.shape[0] == 1:
                        a = a[0]
                    while a.ndim > 2 and a.shape[-1] == 1:
                        a = a[..., 0]
                    if a.shape[0] == 9 and a.shape[1] == hk:
                        a = a.transpose(0, 1)
                    ps.append(a[:hk])
                return torch.stack(ps, dim=0)
            a = qo.float()
            while a.ndim > 3 and a.shape[-1] == 1:
                a = a[..., 0]
            if a.ndim == 3 and a.shape[1] == 9 and a.shape[2] == hk:
                a = a.transpose(1, 2)
            return a[:, :hk, :]

        try:
            h_max = max(hs)
            h_min = min(hs)
            # Single fused forward pass across [main_pos, main_neg, bt0, bt1] when small batch & compatible horizons!
            can_fuse = (K > 1) and (N <= 16) and (h_max <= max(h_min + 64, 2 * h_min))
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                if can_fuse:
                    pv_all = []
                    meta = []
                    for k, (sx, hk, use_sym) in enumerate(zip(slices_x, hs, syms)):
                        x_np = sx.reshape(N, sx.shape[2]).detach().cpu().numpy()
                        s_idx = len(pv_all)
                        pos_k = [torch.from_numpy(_clean_1d_cpu(x_np[j], 8192)).to(device=device, dtype=pt_dtype) for j in range(N)]
                        pv_all.extend(pos_k)
                        if use_sym:
                            pv_all.extend([-t for t in pos_k])
                        meta.append((s_idx, hk, use_sym))
                    qo = self.patchtst(past_values=pv_all, prediction_length=h_max, quantile_levels=DECILES).quantile_outputs
                    qa = _extract_q(qo, h_max)
                    res = []
                    for k, (s_idx, hk, use_sym) in enumerate(meta):
                        if use_sym:
                            qp = qa[s_idx : s_idx + N, :hk]
                            qn = qa[s_idx + N : s_idx + 2 * N, :hk]
                            qk = 0.5 * (qp + (-qn).flip(-1))
                        else:
                            qk = qa[s_idx : s_idx + N, :hk]
                        qk = qk.reshape(B, V, hk, 9)
                        all_nonneg = _nonneg_mask(slices_x[k])
                        qk = torch.where(all_nonneg[..., None, None], qk.clamp_min(0.0), qk)
                        res.append(torch.sort(qk, dim=-1).values)
                    return res

                res = []
                for k, (sx, hk, use_sym) in enumerate(zip(slices_x, hs, syms)):
                    x_np = sx.reshape(N, sx.shape[2]).detach().cpu().numpy()
                    ch_bs = ch_bs_base if (hk <= 256 and sx.shape[2] <= 2048) else ch_bs_big
                    out_chunks = []
                    for i0 in range(0, N, ch_bs):
                        i1 = min(N, i0 + ch_bs)
                        pos_c = [torch.from_numpy(_clean_1d_cpu(x_np[j], 8192)).to(device=device, dtype=pt_dtype) for j in range(i0, i1)]
                        if use_sym:
                            qo = self.patchtst(past_values=pos_c + [-t for t in pos_c], prediction_length=hk, quantile_levels=DECILES).quantile_outputs
                            qa = _extract_q(qo, hk)
                            m_c = i1 - i0
                            out_chunks.append(0.5 * (qa[:m_c] + (-qa[m_c:]).flip(-1)))
                        else:
                            qo = self.patchtst(past_values=pos_c, prediction_length=hk, quantile_levels=DECILES).quantile_outputs
                            out_chunks.append(_extract_q(qo, hk))
                    qk = torch.cat(out_chunks, dim=0).reshape(B, V, hk, 9)
                    all_nonneg = _nonneg_mask(sx)
                    qk = torch.where(all_nonneg[..., None, None], qk.clamp_min(0.0), qk)
                    res.append(torch.sort(qk, dim=-1).values)
                return res
        except Exception:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return [None] * K

    def _build_iamtime_items(self, x, h, past_cov, known_cov):
        B, V, T = x.shape
        t_sl = min(T, 8192)
        s0 = T - t_sl
        x_cpu = x[..., s0:].detach().cpu().float().numpy()
        p_cpu = past_cov[..., s0:].detach().cpu().float().numpy() if past_cov is not None else None
        k_cpu = known_cov[..., s0 : T + h].detach().cpu().float().numpy() if known_cov is not None else None
        mv_max_v = int(self.policy.get("iam_mv_max_v", 24))
        use_mv = bool(self.policy.get("iam_mv_as_cov", True)) and (1 < V <= mv_max_v)
        has_cov = (p_cpu is not None) or (k_cpu is not None) or use_mv
        inps = []
        for b in range(B):
            for v in range(V):
                row = x_cpu[b, v]
                fin = np.flatnonzero(np.isfinite(row))
                st = int(fin[0]) if len(fin) else t_sl - 1
                tgt = row[st:].copy()
                if not has_cov:
                    inps.append({"target": tgt})
                else:
                    pc, fc = {}, {}
                    if use_mv:
                        for ov in range(V):
                            if ov == v: continue
                            arr = x_cpu[b, ov, st:]
                            if np.isfinite(arr).any():
                                pc[f"v{ov}"] = arr
                    if p_cpu is not None:
                        for c in range(p_cpu.shape[1]):
                            arr = p_cpu[b, c, st:]
                            if np.isfinite(arr).any():
                                pc[f"p{c}"] = arr
                    if k_cpu is not None:
                        for c in range(k_cpu.shape[1]):
                            ap = k_cpu[b, c, st:t_sl]
                            af = k_cpu[b, c, t_sl : t_sl + h]
                            if np.isfinite(ap).any() and np.isfinite(af).any():
                                pc[f"k{c}"] = ap; fc[f"k{c}"] = af
                    d = {"target": tgt}
                    if pc: d["past_covariates"] = pc
                    if fc: d["future_covariates"] = fc
                    inps.append(d)
        bs = max(8, 256 // max(1, V)) if use_mv else 64
        return inps, bs

    @torch.inference_mode()
    def _run_iamtime(self, x, h, past_cov, known_cov):
        B, V, T = x.shape
        device = x.device
        try:
            inps, bs = self._build_iamtime_items(x, h, past_cov, known_cov)
            parts = []
            for i0 in range(0, len(inps), bs):
                q, _ = self.iamtime.predict_quantiles(inps[i0 : i0 + bs], prediction_length=h, quantile_levels=DECILES)
                parts.append(q.detach().float())
            q = torch.cat(parts, dim=0).reshape(B, V, h, 9).to(device)
            all_nonneg = _nonneg_mask(x)
            q = torch.where(all_nonneg[..., None, None], q.clamp_min(0.0), q)
            return torch.sort(q, dim=-1).values
        except Exception:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None

    @torch.inference_mode()
    def _run_iamtime_fused_bt(self, slices_x, w_bt, past_covs, known_covs):
        K = len(slices_x)
        B, V = slices_x[0].shape[:2]
        device = slices_x[0].device
        try:
            all_inps = []
            bs = 64
            for k in range(K):
                inps_k, bs_k = self._build_iamtime_items(slices_x[k], w_bt, past_covs[k], known_covs[k])
                all_inps.extend(inps_k)
                bs = min(bs, bs_k)
            parts = []
            for i0 in range(0, len(all_inps), bs):
                q, _ = self.iamtime.predict_quantiles(all_inps[i0 : i0 + bs], prediction_length=w_bt, quantile_levels=DECILES)
                parts.append(q.detach().float())
            q_all = torch.cat(parts, dim=0).reshape(K, B, V, w_bt, 9).to(device)
            res = []
            for k in range(K):
                qk = q_all[k]
                all_nonneg = _nonneg_mask(slices_x[k])
                qk = torch.where(all_nonneg[..., None, None], qk.clamp_min(0.0), qk)
                res.append(torch.sort(qk, dim=-1).values)
            return res
        except Exception:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return [None] * K

    def _safe_fallback(self, q, fb):
        if q is None:
            return fb.clone()
        bad = ~torch.isfinite(q).all(dim=(-2, -1))
        if bad.any():
            q = torch.where(bad[..., None, None], fb, q)
        return q

    def _extract_covs(self, bs, t, h, device, kwargs):
        pasts, knowns = [], []
        for name in ("past_covariates", "categorical_past_covariates", "known_covariates", "categorical_known_covariates"):
            val = kwargs.get(name)
            if val is None: continue
            known = "known" in name
            cov = _as_covariates(val, bs, t + h if known else t, name, device)
            if cov is not None:
                if known: knowns.append(cov)
                else: pasts.append(cov)
        return (torch.cat(pasts, 1) if pasts else None), (torch.cat(knowns, 1) if knowns else None)

    def _embed_21(self, q9, q_t0_21):
        out = q_t0_21.clone()
        idx = DECILE_IDX_21
        for k, pos in enumerate(idx): out[..., pos] = q9[..., k]
        for k in range(8):
            lo_p, hi_p = idx[k], idx[k + 1]
            if hi_p - lo_p == 2: out[..., lo_p + 1] = 0.5 * (q9[..., k] + q9[..., k + 1])
        out[..., 1] = q9[..., 0] + (q_t0_21[..., 1] - q_t0_21[..., idx[0]])
        out[..., 0] = q9[..., 0] + (q_t0_21[..., 0] - q_t0_21[..., idx[0]])
        out[..., -2] = q9[..., -1] + (q_t0_21[..., -2] - q_t0_21[..., idx[-1]])
        out[..., -1] = q9[..., -1] + (q_t0_21[..., -1] - q_t0_21[..., idx[-1]])
        return torch.sort(out, dim=-1).values

    @torch.inference_mode()
    def forecast(self, context, prediction_length, observation_mask=None, cross_learning=None, fast=False, **kwargs):
        x = _as_context(context, observation_mask, device=self.device)
        B, V, T = x.shape
        H = int(prediction_length)
        device = x.device
        N = B * V

        x_flat = x.reshape(N, T)
        obs_f = torch.isfinite(x_flat)
        n_obs = obs_f.sum(dim=-1).float()
        max_n_obs = float(n_obs.max().item()) if N > 0 else 0.0

        x64 = x_flat.double().masked_fill(~obs_f, 0.0)
        mu64 = x64.sum(dim=-1) / n_obs.double().clamp_min(1.0)
        var64 = ((x64 - mu64[:, None]).square() * obs_f.double()).sum(dim=-1) / n_obs.double().clamp_min(1.0)
        loc_t = mu64.float()
        sc_t = var64.sqrt().clamp_min(1e-6).float()

        past_cov, known_cov = self._extract_covs(B, T, H, device, kwargs)
        has_cov = (past_cov is not None) or (known_cov is not None)
        has_known = known_cov is not None

        t0_ctx = int(
            self.policy.get("t0_max_context_known", 24576)
            if has_known
            else self.policy.get("t0_max_context_default", 8192)
        )

        skip_pt = (
            (max_n_obs < float(self.policy.get("skip_pt_below", 64)))
            or (has_known and bool(self.policy.get("skip_pt_on_known_cov", True)))
            or (has_cov and bool(self.policy.get("skip_pt_on_cov", False)))
            or (V > 1 and bool(self.policy.get("skip_pt_on_multivar", False)))
        )

        # Horizon-matched causal backtest cuts (skipped cleanly if fast=True!)
        cuts = []
        w_bt = 0
        if (not fast) and self.policy.get("replay", True):
            h_bt_req = max(H, int(self.policy.get("min_bt_w", 8)))
            if T >= 2048 and H >= 96:
                w_cap = int(self.policy.get("w_cap_xlong", 192))
            elif T >= 1024:
                w_cap = int(self.policy.get("w_cap_long", 96))
            else:
                w_cap = int(self.policy.get("w_cap_short", 48))
            w2 = min(w_cap, h_bt_req, T // 5)
            w1 = min(w_cap, h_bt_req, T // 4)
            if T >= 96 and w2 >= 4 and T >= 2 * w2 + 24:
                w_bt = w2; cuts = [T - 2 * w_bt, T - w_bt]
            elif T >= 48 and w1 >= 4 and T >= w1 + 24:
                w_bt = w1; cuts = [T - w_bt]

        bt_slices = [x[..., :cut] for cut in cuts]
        bt_pcs = [past_cov[..., :cut] if past_cov is not None else None for cut in cuts]
        bt_kcs = [known_cov[..., : cut + w_bt] if known_cov is not None else None for cut in cuts]
        bt_sym = not bool(self.policy.get("fast_bt", True))

        # Execute all 4 orthogonal experts — using parallel CUDA streams when available!
        use_streams = (
            self._is_cuda
            and (self._streams is not None)
            and bool(self.policy.get("parallel_streams", True))
            and (B * V <= 32)  # Biggest win on small-batch latency without increasing peak VRAM on giant batches!
        )

        q_t0_21 = q_t0 = q_ch = q_pt = q_im = None
        bt_t0_list, bt_ch_list, bt_pt_list, bt_im_list = [], [], [], []

        def _do_t0():
            nonlocal q_t0_21, q_t0, bt_t0_list
            want_ctx_gate = (
                bool(cuts)
                and bool(self.policy.get("t0_causal_ctx_gate", False))
                and (T > 8192)
                and (has_known or bool(self.policy.get("t0_ctx_gate_nocov", False)))
            )
            self.last_t0_ctx_16k_frac = 1.0 if (not want_ctx_gate and t0_ctx > 8192 and T > 8192) else 0.0
            if (not want_ctx_gate) and cuts and (not has_cov) and (not bt_sym) and (H <= max(w_bt + 64, 2 * w_bt)) and (B * V <= 16) and (t0_ctx <= 8192):
                # Single fused forward pass: [main, -main, bt0, bt1] at max(H, w_bt) -> 2.5x faster, exact causal prefix!
                h_max = max(H, w_bt)
                self._set_t0_scale_base(min(T, t0_ctx))
                use_amp = self._t0_use_amp(x[..., max(0, T - min(T, t0_ctx)):], min(T, t0_ctx))
                blocks = []
                for sx, sign in [(x, 1.0), (x, -1.0)] + [(bsx, 1.0) for bsx in bt_slices]:
                    tl = sx.shape[2]
                    tsl = min(tl, t0_ctx)
                    x8 = sx[..., tl - tsl :] * sign
                    for i in range(B):
                        row = x8[i]
                        ii = torch.where(row.isfinite().any(0))[0]
                        s = int(ii[0]) if len(ii) else tsl - 1
                        blocks.append(row[:, s:])
                arr, mask, ids = batch_series(blocks)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    out = self.t0.predict(
                        arr.to(device), horizon=h_max, mask=mask.to(device), group_ids=ids.to(device),
                        quantile_levels=QUANTILES_21, trim_leading_missing=True, symmetric_averaging=False,
                        clip_nonnegative=False, snap_integers=False,
                    ).quantiles.float().reshape(2 + len(cuts), B, V, h_max, 21)
                q_t0_21 = 0.5 * (out[0, :, :, :H, :] + (-out[1, :, :, :H, :]).flip(-1))
                all_nn = _nonneg_mask(x)
                q_t0_21 = torch.where(all_nn[..., None, None], q_t0_21.clamp_min(0.0), q_t0_21)
                q_t0_21 = torch.sort(q_t0_21, dim=-1).values
                q_t0 = q_t0_21[..., DECILE_IDX_21]
                bt_t0_list = []
                for ci in range(len(cuts)):
                    qb = out[2 + ci, :, :, :w_bt, :][..., DECILE_IDX_21]
                    nn_c = _nonneg_mask(bt_slices[ci])
                    qb = torch.where(nn_c[..., None, None], qb.clamp_min(0.0), qb)
                    bt_t0_list.append(torch.sort(qb, dim=-1).values)
            elif (
                cuts
                and bool(self.policy.get("t0_causal_ctx_gate", False))
                and (T > 8192)
                and (has_known or bool(self.policy.get("t0_ctx_gate_nocov", False)))
            ):
                # Strictly causal 8K(sb=512) vs 16K(sb=1536) Pop-Quiz context gating!
                ctx_lo = 8192
                ctx_hi = int(self.policy.get("t0_max_context_known", 16384))
                ctx_margin = float(self.policy.get("t0_ctx_bt_margin", -0.005))
                if not has_cov:
                    bt_8k = self._run_t0_fused_bt_nocov(bt_slices, w_bt, max_ctx=ctx_lo, sym=bt_sym)
                    bt_16k = self._run_t0_fused_bt_nocov(bt_slices, w_bt, max_ctx=ctx_hi, sym=bt_sym)
                else:
                    bt_8k = self._run_t0_fused_bt_cov(bt_slices, w_bt, bt_pcs, bt_kcs, max_ctx=ctx_lo, sym=bt_sym)
                    bt_16k = self._run_t0_fused_bt_cov(bt_slices, w_bt, bt_pcs, bt_kcs, max_ctx=ctx_hi, sym=bt_sym)
                q_t_dev = self._q_t.to(device)
                # Score per batch item b across cuts & variates
                l8 = torch.zeros(B, device=device, dtype=torch.float32)
                l16 = torch.zeros(B, device=device, dtype=torch.float32)
                cnt_b = torch.zeros(B, device=device, dtype=torch.float32)
                for ci, cut in enumerate(cuts):
                    yb = x[..., cut : cut + w_bt].reshape(N, w_bt)
                    s8 = _cutoff_mae_crps(bt_8k[ci].reshape(N, w_bt, 9), yb, sc_t, q_t_dev).reshape(B, V, 3)
                    s16 = _cutoff_mae_crps(bt_16k[ci].reshape(N, w_bt, 9), yb, sc_t, q_t_dev).reshape(B, V, 3)
                    ok = (s8[..., 2] * s16[..., 2]) > 0.5
                    ll8 = (0.5 * (s8[..., 0] + 1e-3).log() + 0.5 * (s8[..., 1] + 1e-3).log()) * ok.float()
                    ll16 = (0.5 * (s16[..., 0] + 1e-3).log() + 0.5 * (s16[..., 1] + 1e-3).log()) * ok.float()
                    l8 += ll8.sum(dim=1)
                    l16 += ll16.sum(dim=1)
                    cnt_b += ok.float().sum(dim=1)
                if ctx_margin > 0.0:
                    use16 = (cnt_b == 0) | ((l16 - l8) / cnt_b.clamp_min(1.0) < ctx_margin)
                else:
                    use16 = (cnt_b > 0) & ((l16 - l8) / cnt_b.clamp_min(1.0) < ctx_margin)
                self.last_t0_ctx_16k_frac = float(use16.float().mean().item())
                if bool(use16.all().item()):
                    q_t0_21 = self._run_t0(x, H, past_cov, known_cov, max_ctx=ctx_hi, return_21=True, sym=True)
                    bt_t0_list = bt_16k
                elif not bool(use16.any().item()):
                    q_t0_21 = self._run_t0(x, H, past_cov, known_cov, max_ctx=ctx_lo, return_21=True, sym=True)
                    bt_t0_list = bt_8k
                else:
                    idx16 = torch.where(use16)[0]
                    idx8 = torch.where(~use16)[0]
                    q_t0_21 = torch.empty((B, V, H, 21), device=device, dtype=torch.float32)
                    q_t0_21[idx8] = self._run_t0(
                        x[idx8], H,
                        past_cov[idx8] if past_cov is not None else None,
                        known_cov[idx8] if known_cov is not None else None,
                        max_ctx=ctx_lo, return_21=True, sym=True,
                    )
                    q_t0_21[idx16] = self._run_t0(
                        x[idx16], H,
                        past_cov[idx16] if past_cov is not None else None,
                        known_cov[idx16] if known_cov is not None else None,
                        max_ctx=ctx_hi, return_21=True, sym=True,
                    )
                    bt_t0_list = [
                        torch.where(use16[:, None, None, None], bt_16k[ci], bt_8k[ci])
                        for ci in range(len(cuts))
                    ]
                q_t0 = q_t0_21[..., DECILE_IDX_21]
            else:
                q_t0_21 = self._run_t0(x, H, past_cov, known_cov, max_ctx=t0_ctx, return_21=True, sym=True)
                q_t0 = q_t0_21[..., DECILE_IDX_21]
                if cuts:
                    if not has_cov:
                        bt_t0_list = self._run_t0_fused_bt_nocov(bt_slices, w_bt, max_ctx=t0_ctx, sym=bt_sym)
                    else:
                        bt_t0_list = self._run_t0_fused_bt_cov(bt_slices, w_bt, bt_pcs, bt_kcs, max_ctx=t0_ctx, sym=bt_sym)

        def _do_ch():
            nonlocal q_ch, bt_ch_list
            if cuts and (not has_cov) and (not bt_sym) and (H <= max(w_bt + 64, 2 * w_bt)) and (B * V <= 16):
                # Single fused forward pass: [main, -main, bt0, bt1] at max(H, w_bt) -> 2x faster!
                h_max = max(H, w_bt)
                try:
                    rows = []
                    for sx, sign in [(x, 1.0), (x, -1.0)] + [(bsx, 1.0) for bsx in bt_slices]:
                        tl = sx.shape[2]; tsl = min(tl, 8192)
                        x8 = (sx[..., tl - tsl :] * sign).detach().cpu().float()
                        for i in range(B):
                            row = x8[i]
                            ii = torch.where(row.isfinite().any(0))[0]
                            s = int(ii[0]) if len(ii) else tsl - 1
                            rows.append(row[:, s:])
                    p_all, _ = self.chronos.predict_quantiles(rows, prediction_length=h_max, quantile_levels=DECILES, cross_learning=False)
                    q_all = torch.stack(p_all).float().reshape(2 + len(cuts), B, V, h_max, 9).to(device)
                    q_ch = 0.5 * (q_all[0, :, :, :H, :] + (-q_all[1, :, :, :H, :]).flip(-1))
                    nn_m = _nonneg_mask(x)
                    q_ch = torch.sort(torch.where(nn_m[..., None, None], q_ch.clamp_min(0.0), q_ch), dim=-1).values
                    bt_ch_list = []
                    for ci in range(len(cuts)):
                        qb = q_all[2 + ci, :, :, :w_bt, :]
                        nn_c = _nonneg_mask(bt_slices[ci])
                        bt_ch_list.append(torch.sort(torch.where(nn_c[..., None, None], qb.clamp_min(0.0), qb), dim=-1).values)
                except Exception:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    q_ch = None
                    bt_ch_list = [None] * len(cuts)
            else:
                q_ch = self._run_chronos(x, H, past_cov, known_cov, fallback=None, sym=True)
                if cuts:
                    if not has_cov and len(cuts) == 2:
                        bt_ch_list = self._run_chronos_fused_bt_nocov(bt_slices, w_bt, sym=bt_sym)
                    else:
                        bt_ch_list = [
                            self._run_chronos(bt_slices[ci], w_bt, bt_pcs[ci], bt_kcs[ci], fallback=None, sym=bt_sym)
                            for ci in range(len(cuts))
                        ]

        def _do_pt():
            nonlocal q_pt, bt_pt_list
            if skip_pt:
                q_pt = None
                bt_pt_list = [None] * len(cuts)
            else:
                all_s = [x] + bt_slices
                all_h = [H] + [w_bt] * len(cuts)
                all_sym = [True] + [bt_sym] * len(cuts)
                res_pt = self._run_patchtst_slices(all_s, all_h, syms=all_sym)
                q_pt = res_pt[0]
                bt_pt_list = res_pt[1:]

        def _do_im():
            nonlocal q_im, bt_im_list
            if cuts and (B * V <= 16) and (((not has_known) and (H <= max(w_bt + 64, 2 * w_bt))) or (has_known and (H == w_bt))):
                # Single fused iAmTime forward pass [main, bt0, bt1]: ~2x faster & 100% strictly causal!
                h_max = max(H, w_bt)
                all_sx = [x] + bt_slices
                all_pc = [past_cov] + bt_pcs
                all_kc = ([known_cov] + bt_kcs) if has_known else ([None] * (1 + len(cuts)))
                res_im = self._run_iamtime_fused_bt(all_sx, h_max, all_pc, all_kc)
                q_im = res_im[0][:, :, :H, :] if res_im[0] is not None else None
                bt_im_list = [r[:, :, :w_bt, :] if r is not None else None for r in res_im[1:]]
            else:
                q_im = self._run_iamtime(x, H, past_cov, known_cov)
                if cuts:
                    bt_im_list = self._run_iamtime_fused_bt(bt_slices, w_bt, bt_pcs, bt_kcs)

        if use_streams:
            cur = torch.cuda.current_stream(device)
            ev = torch.cuda.Event()
            ev.record(cur)
            for s, fn in zip(self._streams, (_do_t0, _do_ch, _do_pt, _do_im)):
                s.wait_event(ev)
                with torch.cuda.stream(s):
                    fn()
            for s in self._streams:
                cur.wait_stream(s)
        else:
            _do_t0(); _do_ch(); _do_pt(); _do_im()

        q_ch = self._safe_fallback(q_ch, q_t0)
        q_pt = q_t0.clone() if skip_pt else self._safe_fallback(q_pt, q_t0)
        q_im = self._safe_fallback(q_im, q_t0)
        if cuts:
            bt_ch_list = [self._safe_fallback(bt_ch_list[ci], bt_t0_list[ci]) for ci in range(len(cuts))]
            bt_pt_list = [bt_t0_list[ci].clone() if skip_pt else self._safe_fallback(bt_pt_list[ci], bt_t0_list[ci]) for ci in range(len(cuts))]
            bt_im_list = [self._safe_fallback(bt_im_list[ci], bt_t0_list[ci]) for ci in range(len(cuts))]

        # Independent Per-Expert Causal Pop-Quiz Self-Gated Cross-Inference (short univariate panels, no cov)
        ci_mode = False if fast else (self.policy.get("cross_learning", "auto") if cross_learning is None else cross_learning)
        ci_short_t = int(self.policy.get("ci_short_t", 0))
        ci_short_both = bool(self.policy.get("ci_short_both", False))
        ci_max_t = int(self.policy.get("ci_max_t", 1024))
        ci_max_grp = int(self.policy.get("ci_max_group", 64))
        ci_margin = float(self.policy.get("ci_bt_margin", -0.012))
        use_t0_ci, use_ch_ci = False, False

        if ci_mode and (V == 1) and (B > 1) and (not has_cov) and (T <= ci_max_t):
            if ci_mode is True:
                use_t0_ci, use_ch_ci = True, True
                q_t0_21 = self._run_t0_ci(x, H, return_21=True, max_grp=ci_max_grp, sym=True)
                q_t0 = q_t0_21[..., DECILE_IDX_21]
                q_ch = self._run_chronos_ci(x, H, q_t0, max_grp=ci_max_grp, sym=True)
                if cuts:
                    bt_t0_list = [self._run_t0_ci(sx, w_bt, return_21=False, max_grp=ci_max_grp, sym=bt_sym) for sx in bt_slices]
                    bt_ch_list = [self._run_chronos_ci(sx, w_bt, bt_t0_list[ci], max_grp=ci_max_grp, sym=bt_sym) for ci, sx in enumerate(bt_slices)]
            elif ci_mode == "auto":
                short_bayesian = (ci_short_t > 0) and (T <= ci_short_t) and (B >= 4)
                q_t_dev = self._q_t.to(device)
                if cuts:
                    ci_cuts, ci_w, ci_slices = cuts, w_bt, bt_slices
                    ind_t0, ind_ch = bt_t0_list, bt_ch_list
                elif T >= 18:
                    ci_w = max(4, min(H, T // 3))
                    ci_cuts = [T - ci_w]
                    ci_slices = [x[..., : ci_cuts[0]]]
                    ind_t0 = [self._run_t0(ci_slices[0], ci_w, None, None, max_ctx=8192, return_21=False, sym=bt_sym)]
                    ind_ch = [self._safe_fallback(self._run_chronos(ci_slices[0], ci_w, None, None, fallback=None, sym=bt_sym), ind_t0[0])]
                else:
                    ci_cuts = []

                if short_bayesian:
                    use_t0_ci = True
                    q_t0_21 = self._run_t0_ci(x, H, return_21=True, max_grp=ci_max_grp, sym=True)
                    q_t0 = q_t0_21[..., DECILE_IDX_21]
                    if cuts:
                        bt_t0_list = [self._run_t0_ci(sx, w_bt, return_21=False, max_grp=ci_max_grp, sym=bt_sym) for sx in bt_slices]
                    if ci_short_both:
                        use_ch_ci = True
                        q_ch = self._run_chronos_ci(x, H, q_t0, max_grp=ci_max_grp, sym=True)
                        if cuts:
                            bt_ch_list = [self._run_chronos_ci(sx, w_bt, bt_t0_list[ci], max_grp=ci_max_grp, sym=bt_sym) for ci, sx in enumerate(bt_slices)]
                    elif ci_cuts:
                        bt_ch_ci = [self._run_chronos_ci(sx, ci_w, ind_t0[ci], max_grp=ci_max_grp, sym=bt_sym) for ci, sx in enumerate(ci_slices)]
                        dch_vals = []
                        for ci, cut in enumerate(ci_cuts):
                            yb = x[:, 0, cut : cut + ci_w]
                            si = _cutoff_mae_crps(ind_ch[ci][:, 0], yb, sc_t, q_t_dev)
                            sc = _cutoff_mae_crps(bt_ch_ci[ci][:, 0], yb, sc_t, q_t_dev)
                            ok = (si[:, 2] * sc[:, 2]) > 0.5
                            if ok.any():
                                li = (0.5 * (si[ok, 0] + 1e-3).log() + 0.5 * (si[ok, 1] + 1e-3).log()).mean().item()
                                lc = (0.5 * (sc[ok, 0] + 1e-3).log() + 0.5 * (sc[ok, 1] + 1e-3).log()).mean().item()
                                dch_vals.append(lc - li)
                        if dch_vals and (float(np.mean(dch_vals)) < ci_margin):
                            use_ch_ci = True
                            q_ch = self._run_chronos_ci(x, H, q_t0, max_grp=ci_max_grp, sym=True)
                            if cuts: bt_ch_list = bt_ch_ci
                elif ci_cuts:
                    bt_t0_ci = [self._run_t0_ci(sx, ci_w, return_21=False, max_grp=ci_max_grp, sym=bt_sym) for sx in ci_slices]
                    bt_ch_ci = [self._run_chronos_ci(sx, ci_w, bt_t0_ci[ci], max_grp=ci_max_grp, sym=bt_sym) for ci, sx in enumerate(ci_slices)]
                    dt0_vals, dch_vals = [], []
                    for ci, cut in enumerate(ci_cuts):
                        yb = x[:, 0, cut : cut + ci_w]
                        for q_ind, q_ci, acc in [(ind_t0[ci][:, 0], bt_t0_ci[ci][:, 0], dt0_vals), (ind_ch[ci][:, 0], bt_ch_ci[ci][:, 0], dch_vals)]:
                            si = _cutoff_mae_crps(q_ind, yb, sc_t, q_t_dev)
                            sc = _cutoff_mae_crps(q_ci, yb, sc_t, q_t_dev)
                            ok = (si[:, 2] * sc[:, 2]) > 0.5
                            if ok.any():
                                li = (0.5 * (si[ok, 0] + 1e-3).log() + 0.5 * (si[ok, 1] + 1e-3).log()).mean().item()
                                lc = (0.5 * (sc[ok, 0] + 1e-3).log() + 0.5 * (sc[ok, 1] + 1e-3).log()).mean().item()
                                acc.append(lc - li)
                    if dt0_vals and (float(np.mean(dt0_vals)) < ci_margin):
                        use_t0_ci = True
                        q_t0_21 = self._run_t0_ci(x, H, return_21=True, max_grp=ci_max_grp, sym=True)
                        q_t0 = q_t0_21[..., DECILE_IDX_21]
                        if cuts: bt_t0_list = bt_t0_ci
                    if dch_vals and (float(np.mean(dch_vals)) < ci_margin):
                        use_ch_ci = True
                        q_ch = self._run_chronos_ci(x, H, q_t0, max_grp=ci_max_grp, sym=True)
                        if cuts: bt_ch_list = bt_ch_ci
                else:
                    use_t0_ci = True
                    q_t0_21 = self._run_t0_ci(x, H, return_21=True, max_grp=ci_max_grp, sym=True)
                    q_t0 = q_t0_21[..., DECILE_IDX_21]

        self.last_cross_learning_used = {"t0": use_t0_ci, "chronos2": use_ch_ci}

        # 4-architecture regime-aware prior: [t0-Dyn32K, chronos2, patchtst, iamtime]
        if has_known:
            pr_long = self.policy.get("prior_cov_long", [0.48, 0.20, 0.00, 0.32])
            pr_short = self.policy.get("prior_cov_short", [0.48, 0.24, 0.00, 0.28])
        elif has_cov:
            pr_long = self.policy.get("prior_pcov_long", [0.54, 0.20, 0.04, 0.22])
            pr_short = self.policy.get("prior_pcov_short", [0.56, 0.22, 0.02, 0.20])
        elif V > 1:
            pr_long = self.policy.get("prior_mv_long", [0.50, 0.20, 0.10, 0.20])
            pr_short = self.policy.get("prior_mv_short", [0.54, 0.24, 0.02, 0.20])
        else:
            pr_long = self.policy.get("prior_long", [0.42, 0.16, 0.30, 0.12])
            pr_short = self.policy.get("prior_short", [0.58, 0.24, 0.02, 0.16])

        p_l = torch.tensor(pr_long, device=device, dtype=torch.float32)
        p_s = torch.tensor(pr_short, device=device, dtype=torch.float32)
        s_lo = float(self.policy.get("short_lo", 80))
        s_hi = float(self.policy.get("short_hi", 160))
        alpha_l = ((n_obs - s_lo) / max(1.0, s_hi - s_lo)).clamp(0.0, 1.0)[:, None]
        prior = ((1.0 - alpha_l) * p_s[None, :] + alpha_l * p_l[None, :]).clamp_min(1e-4)

        # Monotonic/cumulative curve guard for arcsinh-based iAmTime (general signal property)
        mono_damp = float(self.policy.get("iam_monotone_damp", 0.15))
        if mono_damp < 1.0:
            mono = _monotone_mask(x).reshape(N)
            if mono.any():
                prior = prior.clone()
                prior[mono, 3] = (prior[mono, 3] * mono_damp).clamp_min(1e-4)
                prior = prior / prior.sum(dim=-1, keepdim=True)

        if skip_pt:
            prior[:, 2] = 1e-4
            prior = prior / prior.sum(dim=-1, keepdim=True)
        log_prior = prior.log()

        if cuts:
            bt = torch.zeros((N, 2, 4, 3), device=device, dtype=torch.float32)
            q_t = self._q_t.to(device)
            for ci, cut in enumerate(cuts):
                slot = (2 - len(cuts)) + ci
                y_bt = x[..., cut : cut + w_bt].reshape(N, w_bt)
                bt[:, slot, 0] = _cutoff_mae_crps(bt_t0_list[ci].reshape(N, w_bt, 9), y_bt, sc_t, q_t)
                bt[:, slot, 1] = _cutoff_mae_crps(bt_ch_list[ci].reshape(N, w_bt, 9), y_bt, sc_t, q_t)
                bt[:, slot, 2] = _cutoff_mae_crps(bt_pt_list[ci].reshape(N, w_bt, 9), y_bt, sc_t, q_t)
                bt[:, slot, 3] = _cutoff_mae_crps(bt_im_list[ci].reshape(N, w_bt, 9), y_bt, sc_t, q_t)

            mae = bt[..., 0].clamp_min(0.0)
            crps = bt[..., 1].clamp_min(0.0)
            ok_all = bt[..., 2].clamp(0.0, 1.0).min(dim=-1, keepdim=True).values
            ll = 0.50 * (mae + 1e-3).log() + 0.50 * (crps + 1e-3).log()
            active = [0, 1, 3] if skip_pt else [0, 1, 2, 3]
            if self.policy.get("ref_mode", "median") == "median":
                ref = ll[..., active].median(dim=-1, keepdim=True).values
            else:
                ref = ll[..., active].mean(dim=-1, keepdim=True)
            rel_l = (ll - ref).clamp(-1.2, 1.2) * ok_all
            cnt = ok_all.sum(dim=1)
            n_eff = cnt * float(w_bt)
            shrink = n_eff / (n_eff + float(self.policy.get("kappa", 16.0)))
            both = ok_all[:, 0] * ok_all[:, 1]
            disagree = (rel_l[:, 0] * rel_l[:, 1] < 0).float() * both
            dis_pen = float(self.policy.get("disagree_pen", 0.40))
            tilt = shrink * (1.0 - dis_pen * disagree) * (rel_l.sum(dim=1) / cnt.clamp_min(1.0))
            logits = log_prior - float(self.policy.get("tau", 11.0)) * tilt
            if skip_pt:
                logits[:, 2] = -1e9
        else:
            logits = log_prior
            if skip_pt:
                logits[:, 2] = -1e9

        w_post = torch.softmax(logits, dim=-1)
        self.last_routing_weights = w_post.detach().cpu().numpy()
        self.last_expert_medians = {
            "t0": q_t0[..., 4].detach().cpu().numpy(),
            "chronos2": q_ch[..., 4].detach().cpu().numpy(),
            "patchtst": q_pt[..., 4].detach().cpu().numpy(),
            "iamtime": q_im[..., 4].detach().cpu().numpy(),
        }

        w = w_post[:, None, :].expand(N, H, 4)
        qs = torch.stack(
            [q_t0.reshape(N, H, 9), q_ch.reshape(N, H, 9), q_pt.reshape(N, H, 9), q_im.reshape(N, H, 9)],
            dim=1,
        )
        qn = ((qs - loc_t[:, None, None, None]) / sc_t[:, None, None, None]).clamp(-1e3, 1e3)
        qp = torch.sort(qn.permute(0, 2, 1, 3), dim=-1).values
        if self.policy.get("mixture_cdf", True):
            q_comb = (mixture_cdf_quantiles(qp, w) * sc_t[:, None, None] + loc_t[:, None, None]).reshape(B, V, H, 9)
        else:
            q_comb = ((qp * w.unsqueeze(-1)).sum(dim=2) * sc_t[:, None, None] + loc_t[:, None, None]).reshape(B, V, H, 9)

        all_nonneg = _nonneg_mask(x)
        if self.policy.get("nonneg_clip", True):
            q_comb = torch.where(all_nonneg[..., None, None], q_comb.clamp_min(0.0), q_comb)

        if self.policy.get("integer_snap", True) and T >= 84:
            tail = x[..., -84:]
            elig = (torch.isfinite(tail) & (tail >= 0) & (tail == tail.round())).all(dim=-1)
            if elig.any():
                q_comb = torch.where(elig[..., None, None], q_comb.round(), q_comb)

        q_comb = torch.sort(q_comb, dim=-1).values
        q21 = self._embed_21(q_comb, q_t0_21)
        if self.policy.get("nonneg_clip", True):
            q21 = torch.where(all_nonneg[..., None, None], q21.clamp_min(0.0), q21)
        q21 = torch.sort(q21, dim=-1).values
        return ZitraV3ForecastOutput(quantiles=q21, quantile_levels=QUANTILES_21)
