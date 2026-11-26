"""Zitra-Core-v3 (100% Pure Apache-2.0): Single-Backbone Distilled 256M Probabilistic Forecaster.

Single pure 255.6M Patch-Decoder Transformer (~511 MB bf16 on disk, zero ensembling at inference),
distilled from the 4-architecture Zitra-Titan-v3 teacher:
  - Precision Guard: exact fp32 (TF32 on CUDA) on T<=2048 or high-offset cumulative/macro series
    (eliminating bfloat16 mantissa truncation on large cumulative curves); bf16 autocast on T>2048
  - Fused single-pass [x, -x] sign-equivariant mirroring (or single-pass `fast=True`)
  - Native 8,192 context (exact scale_base=512 RoPE) with automatic Dynamic-RoPE (1536/2048) if T>8192
  - Causal Self-Gated Cross-Learning (`cross_learning="auto"`, works down to T>=12, or explicit True/False)
  - Native known-future & past covariates, 21-quantile grid (0.01..0.99), nonneg clip & integer snap
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from t0 import MaskType, T0Forecaster, TimeSeries, VariateType, batch_series

DECILES = [i / 10.0 for i in range(1, 10)]
QUANTILES_21 = [
    0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
    0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99,
]
DECILE_IDX_21 = [2, 4, 6, 8, 10, 12, 14, 16, 18]


@dataclass
class ZitraCoreForecastOutput:
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
    if isinstance(val, dict):
        if not val:
            return None
        parts = []
        for k in sorted(val.keys()):
            a = val[k]
            t = a if isinstance(a, torch.Tensor) else torch.as_tensor(a, dtype=torch.float32)
            t = t.to(device=device, dtype=torch.float32)
            if t.ndim == 1:
                t = t[None, None, :]
            elif t.ndim == 2:
                t = t[:, None, :]
            parts.append(t)
        c = torch.cat(parts, dim=1)
    else:
        c = val if isinstance(val, torch.Tensor) else torch.as_tensor(val, dtype=torch.float32)
        c = c.to(device=device, dtype=torch.float32)
        if c.ndim == 1:
            c = c[None, None, :]
        elif c.ndim == 2:
            c = c[:, None, :]
    if c.shape[0] != bs:
        raise ValueError(f"{name} batch {c.shape[0]} != {bs}")
    if c.shape[2] < exp_len:
        pad = torch.full((bs, c.shape[1], exp_len - c.shape[2]), float("nan"), device=device, dtype=torch.float32)
        c = torch.cat([c, pad], dim=2)
    elif c.shape[2] > exp_len:
        c = c[..., :exp_len]
    return c


def _nonneg_mask(x: torch.Tensor) -> torch.Tensor:
    fin = torch.isfinite(x)
    has = fin.any(dim=-1)
    xmin = torch.where(fin, x, torch.full_like(x, float("inf"))).min(dim=-1).values
    return has & (xmin >= 0.0)


def _integer_mask(x: torch.Tensor) -> torch.Tensor:
    fin = torch.isfinite(x)
    n_fin = fin.sum(dim=-1)
    xv = torch.where(fin, x, torch.zeros_like(x))
    is_int = ((xv - xv.round()).abs() <= 1e-4) | (~fin)
    return (n_fin >= 4) & is_int.all(dim=-1)


def _needs_fp32(x: torch.Tensor, t_sl: int, min_bf16_t: int = 2048) -> bool:
    """Use exact fp32 (TF32 on CUDA) on short/medium series or high offset-to-variation series."""
    if t_sl <= min_bf16_t:
        return True
    fin = torch.isfinite(x)
    cnt = fin.float().sum(dim=-1).clamp_min(1.0)
    xv = torch.where(fin, x, torch.zeros_like(x))
    mu = xv.sum(dim=-1) / cnt
    sd = (((xv - mu[..., None]).pow(2) * fin.float()).sum(dim=-1) / cnt).sqrt().clamp_min(1e-6)
    ratio = (mu.abs() / sd).max().item()
    return ratio > 30.0


def _cutoff_mae_crps(q_pred, y_true, scale, q_levels):
    valid = torch.isfinite(y_true)
    cnt = valid.sum(dim=-1)
    ok = cnt >= 1
    yv = torch.where(valid, y_true, torch.zeros_like(y_true))
    q5 = q_pred[..., 4]
    mae = ((q5 - yv).abs() * valid).sum(dim=-1) / cnt.clamp_min(1) / scale.clamp_min(1e-6)
    diff = yv[..., None] - q_pred
    pin = 2.0 * torch.maximum(q_levels * diff, (q_levels - 1.0) * diff).mean(dim=-1)
    crps = (pin * valid).sum(dim=-1) / cnt.clamp_min(1) / scale.clamp_min(1e-6)
    return torch.stack([mae, crps, ok.float()], dim=-1)


class ZitraCoreV3Forecaster:
    def __init__(self, t0_model, policy=None):
        self.t0 = t0_model
        self.policy = policy or {}
        self.device = next(t0_model.parameters()).device
        self.max_context = int(self.policy.get("max_context", 8192))
        self.quantile_levels = QUANTILES_21
        self._q_t = torch.tensor(DECILES, dtype=torch.float32)
        self.last_routing_weights = None
        self.last_expert_medians = None
        self.last_cross_learning_used = {"t0": False}
        self._is_cuda = torch.cuda.is_available() and str(self.device).startswith("cuda")
        if self._is_cuda:
            torch.set_float32_matmul_precision("high")

    @classmethod
    def from_pretrained(cls, path, device="cuda"):
        pkg = Path(path)
        policy_p = pkg / "policy.json"
        policy = json.loads(policy_p.read_text()) if policy_p.exists() else {}
        cfg_p = pkg / "t0_config.json"
        if not cfg_p.exists():
            cfg_p = pkg / "config.json"
        t0_cfg = json.loads(cfg_p.read_text())
        sd = load_file(str(pkg / "model.safetensors"))
        if any(k.startswith("t0.") for k in sd.keys()):
            sd = {k[3:]: v.float() for k, v in sd.items() if k.startswith("t0.")}
        else:
            sd = {k: v.float() for k, v in sd.items()}
        t0_model = T0Forecaster(**t0_cfg)
        t0_model.load_state_dict(sd, strict=True)
        t0_model.transformer.rotary_emb.scale_base = 512.0
        t0_model = t0_model.to(device=device, dtype=torch.float32).eval()
        return cls(t0_model, policy=policy)

    def _set_t0_scale_base(self, t_sl: int):
        if t_sl <= 8192:
            self.t0.transformer.rotary_emb.scale_base = 512.0
        elif t_sl <= 16384:
            self.t0.transformer.rotary_emb.scale_base = 1536.0
        else:
            self.t0.transformer.rotary_emb.scale_base = 2048.0

    @torch.inference_mode()
    def _run_t0(self, x, h, past_cov, known_cov, max_ctx=8192, return_21=True, sym=True):
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
        min_bf16_t = int(self.policy.get("bf16_min_t", 2048))
        use_amp = self._is_cuda and bool(self.policy.get("t0_bf16_amp", True)) and (not _needs_fp32(x8, t_sl, min_bf16_t))

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
                            parts_b.append(out)
                        else:
                            arr, mask, ids = batch_series(blocks)
                            out = self.t0.predict(
                                arr.to(device), horizon=h, mask=mask.to(device), group_ids=ids.to(device),
                                quantile_levels=q_req, trim_leading_missing=True, symmetric_averaging=sym,
                                clip_nonnegative=False, snap_integers=False,
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
                            clip_nonnegative=False, snap_integers=False,
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
    def _run_t0_ci(self, x, h, return_21=True, max_grp=64, sym=True):
        B, V, T = x.shape
        device = x.device
        t_sl = min(T, 8192)
        self._set_t0_scale_base(t_sl)
        x8 = x[..., T - t_sl :]
        q_req = QUANTILES_21 if return_21 else DECILES
        n_q = len(q_req)
        min_bf16_t = int(self.policy.get("bf16_min_t", 2048))
        use_amp = self._is_cuda and bool(self.policy.get("t0_bf16_amp", True)) and (not _needs_fp32(x8, t_sl, min_bf16_t))
        parts = []
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            for i0 in range(0, B, max_grp):
                sub = x8[i0 : i0 + max_grp, 0, :]
                ii = torch.where(sub.isfinite().any(0))[0]
                s = int(ii[0]) if len(ii) else t_sl - 1
                arr, mask, ids = batch_series([sub[:, s:]])
                out = self.t0.predict(
                    arr.to(device), horizon=h, mask=mask.to(device), group_ids=ids.to(device),
                    quantile_levels=q_req, trim_leading_missing=True, symmetric_averaging=sym,
                    clip_nonnegative=False, snap_integers=False,
                ).quantiles.float()
                parts.append(out.reshape(sub.shape[0], 1, h, n_q))
        return torch.cat(parts, dim=0)

    def _extract_covs(self, bs, t, h, device, kwargs):
        pc_list, kc_list = [], []
        for k in ("past_covariates", "categorical_past_covariates"):
            c = _as_covariates(kwargs.get(k), bs, t, k, device)
            if c is not None:
                pc_list.append(c)
        for k in ("known_covariates", "categorical_known_covariates"):
            c = _as_covariates(kwargs.get(k), bs, t + h, k, device)
            if c is not None:
                kc_list.append(c)
        past_cov = torch.cat(pc_list, dim=1) if pc_list else None
        known_cov = torch.cat(kc_list, dim=1) if kc_list else None
        return past_cov, known_cov

    @torch.inference_mode()
    def forecast(
        self,
        context,
        prediction_length: int,
        observation_mask=None,
        cross_learning=None,
        fast: bool = False,
        **kwargs,
    ):
        H = int(prediction_length)
        x = _as_context(context, observation_mask=observation_mask, device=self.device)
        B, V, T = x.shape
        N = B * V
        past_cov, known_cov = self._extract_covs(B, T, H, self.device, kwargs)
        has_cov = (past_cov is not None) or (known_cov is not None)
        t0_ctx = int(self.policy.get("max_context", 8192))
        sym = not bool(fast)

        cl_mode = self.policy.get("cross_learning", "auto") if cross_learning is None else cross_learning
        ci_max_t = int(self.policy.get("ci_max_t", 2048))
        ci_max_grp = int(self.policy.get("ci_max_group", 64))
        ci_margin = float(self.policy.get("ci_bt_margin", -0.005))
        use_ci = False

        can_ci = (not fast) and (not has_cov) and (V == 1) and (B > 1) and (T <= ci_max_t) and (cl_mode in ("auto", True))
        if can_ci and cl_mode is True:
            use_ci = True
            q21 = self._run_t0_ci(x, H, return_21=True, max_grp=ci_max_grp, sym=sym)
        elif can_ci and cl_mode == "auto":
            ci_short_t = int(self.policy.get("ci_short_t", 96))
            if T <= ci_short_t and B >= 4:
                # Short panel Bayesian shrinkage: individual history is short (<96 steps), so panel cross-learning dominates
                use_ci = True
            else:
                w_bt = min(H, max(2, T // 4), 48)
                cut = T - w_bt
                if cut >= 8:
                    x_bt = x[..., :cut]
                    y_bt = x[..., cut : cut + w_bt].reshape(N, w_bt)
                    fin = torch.isfinite(x_bt)
                    xv = torch.where(fin, x_bt, torch.zeros_like(x_bt))
                    cnt_c = fin.float().sum(dim=-1).clamp_min(1.0)
                    mu = xv.sum(dim=-1) / cnt_c
                    sd = (((xv - mu[..., None]).pow(2) * fin.float()).sum(dim=-1) / cnt_c).sqrt()
                    dx = (x_bt[..., 1:] - x_bt[..., :-1]).abs()
                    dfin = torch.isfinite(dx)
                    mad = (dx.nan_to_num(0.0) * dfin.float()).sum(dim=-1) / dfin.float().sum(dim=-1).clamp_min(1.0)
                    sc_t = torch.maximum(mad, 0.1 * sd).clamp_min(1e-3).reshape(N)
                    q_t = self._q_t.to(self.device)
                    qi_bt = self._run_t0(x_bt, w_bt, None, None, max_ctx=t0_ctx, return_21=False, sym=False).reshape(N, w_bt, 9)
                    qc_bt = self._run_t0_ci(x_bt, w_bt, return_21=False, max_grp=ci_max_grp, sym=False).reshape(N, w_bt, 9)
                    si = _cutoff_mae_crps(qi_bt, y_bt, sc_t, q_t)
                    sc = _cutoff_mae_crps(qc_bt, y_bt, sc_t, q_t)
                    ok = (si[:, 2] > 0.5) & (sc[:, 2] > 0.5)
                    if ok.any():
                        li = (0.5 * (si[ok, 0] + 1e-3).log() + 0.5 * (si[ok, 1] + 1e-3).log()).mean().item()
                        lc = (0.5 * (sc[ok, 0] + 1e-3).log() + 0.5 * (sc[ok, 1] + 1e-3).log()).mean().item()
                        if (lc - li) < ci_margin:
                            use_ci = True
            if use_ci:
                q21 = self._run_t0_ci(x, H, return_21=True, max_grp=ci_max_grp, sym=sym)
            else:
                q21 = self._run_t0(x, H, past_cov, known_cov, max_ctx=t0_ctx, return_21=True, sym=sym)
        else:
            q21 = self._run_t0(x, H, past_cov, known_cov, max_ctx=t0_ctx, return_21=True, sym=sym)

        self.last_cross_learning_used = {"t0": use_ci}

        if self.policy.get("nonneg_clip", True):
            nn_mask = _nonneg_mask(x)
            q21 = torch.where(nn_mask[..., None, None], q21.clamp_min(0.0), q21)
        if self.policy.get("integer_snap", True):
            int_mask = _integer_mask(x)
            q21 = torch.where(int_mask[..., None, None], q21.round(), q21)
        q21 = torch.sort(q21, dim=-1).values
        return ZitraCoreForecastOutput(quantiles=q21, quantile_levels=QUANTILES_21)


# Backwards-compatible alias
Zitra256MForecaster = ZitraCoreV3Forecaster
