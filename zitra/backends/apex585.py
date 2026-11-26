"""Zitra-Apex-585M: Self-contained single-checkpoint Multi-Prior Probabilistic Forecaster.
Combines 3 orthogonal commercially-permissive zero-overlap experts:
  1. Native 209M (TimestralModel / Zitra-209M-v2, decoder + cross-variate RoPE)
  2. T0-gamma 256M (Apache-2.0, full a2 mode: trim_leading_missing + sym_avg + nonneg + int_snap + 8192 ctx)
  3. Chronos-2 120M (Apache-2.0, encoder-only, a2 sym_avg + nonneg + native multivariate & covariate support)
via an Unclamped Neural Meta-Combiner (zero_replay / one_step / InfiLoop loop3) with
full-simplex median+spread routing, robust outlier-immune 3-expert median gating,
and optional conditional causal backtest."""
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from safetensors.torch import load_file
from timestral import TimestralForecaster
from timestral.config import TimestralConfig
from timestral.model import TimestralModel
from t0 import T0Forecaster, TimeSeries, batch_series
from chronos.chronos2 import Chronos2Pipeline, Chronos2Model, Chronos2CoreConfig

DECILES = [i / 10.0 for i in range(1, 10)]
PRIOR_LOGITS = [math.log(0.65), math.log(0.23), math.log(0.12)]


def build_meta_features(q_norm, feats, bt, use_backtest=True):
    """
    q_norm: (B, 3, H, 9) normalized quantiles
    feats:  (B, 14) history summary features
    bt:     (B, 2, 3, 5) causal backtest stats [mae, crps, bias, cov80, ok]
    Returns: per-horizon feature tensor (B, H, F_dim)
    """
    B, E, H, Q = q_norm.shape
    assert E == 3 and Q == 9
    med = q_norm[..., 4]  # (B, 3, H)
    lo = (med - q_norm[..., 1]).clamp_min(1e-5)  # med - q20
    hi = (q_norm[..., 7] - med).clamp_min(1e-5)  # q80 - med
    pool_med = med.median(dim=1, keepdim=True).values  # (B, 1, H)
    pos = torch.arange(1, H + 1, device=q_norm.device, dtype=q_norm.dtype).log1p() / 6.0
    pos_e = pos[None, None, :].expand(B, E, H)

    # Per-expert features (B, H, 3*5=15)
    exp_f = torch.stack([
        med.asinh(),
        (lo + hi).log(),
        (hi / lo).log(),
        (med - pool_med).asinh(),
        pos_e,
    ], dim=-1).clamp(-8.0, 8.0)  # (B, 3, H, 5)
    exp_f = exp_f.permute(0, 2, 1, 3).reshape(B, H, E * 5)

    # Cross-expert disagreement features (B, H, 6)
    m0, m1, m2 = med[:, 0], med[:, 1], med[:, 2]
    spr = (lo + hi)
    dis = torch.stack([
        (m1 - m0).asinh(),
        (m2 - m0).asinh(),
        (m1 - m2).asinh(),
        med.std(dim=1).log1p(),
        spr.std(dim=1).log1p(),
        pos[None, :].expand(B, H),
    ], dim=-1).clamp(-8.0, 8.0)  # (B, H, 6)

    # History features broadcast across horizon (B, H, 14)
    h_f = feats[:, None, :].expand(B, H, -1)

    if not use_backtest:
        return torch.cat([exp_f, dis, h_f], dim=-1)  # 15 + 6 + 14 = 35

    # Rich causal backtest features (BOTH cutoffs + mean + cross-cutoff consistency t-stat!)
    # bt: (B, 2, 3, 5) -> mae, crps, bias, cov, ok
    mae = bt[..., 0].clamp_min(0.0)   # (B, 2, 3)
    crps = bt[..., 1].clamp_min(0.0)
    bias = bt[..., 2]
    cov = bt[..., 3]
    ok = bt[..., 4]  # (B, 2, 3)
    loss = 0.5 * (mae + crps) + 1e-4  # (B, 2, 3)
    # Relative log loss vs Timestral (expert 0) at each cutoff
    rel_l = (loss.log() - loss[..., :1].log()).clamp(-3.0, 3.0) * ok  # (B, 2, 3)
    rel_mae = ((mae + 1e-4).log() - (mae[..., :1] + 1e-4).log()).clamp(-3.0, 3.0) * ok
    rel_crps = ((crps + 1e-4).log() - (crps[..., :1] + 1e-4).log()).clamp(-3.0, 3.0) * ok
    # Cross-cutoff mean & consistency t-stat (exact same signal available to 1-step & looped!)
    ok_sum = ok.sum(dim=1).clamp_min(1.0)  # (B, 3)
    mean_rel = rel_l.sum(dim=1) / ok_sum
    diff_rel = (rel_l[:, 1] - rel_l[:, 0]).abs() * (ok[:, 0] * ok[:, 1])
    tstat = (mean_rel / (diff_rel + 0.20)).clamp(-4.0, 4.0)
    # Flatten per-series bt features:
    # rel_l(6) + rel_mae(6) + rel_crps(6) + mean_rel(3) + tstat(3) + log1p(loss)(6) + asinh(bias)(6) + (cov-0.8)(6) + ok(6) = 48
    bt_vec = torch.cat([
        rel_l.reshape(B, 6),
        rel_mae.reshape(B, 6),
        rel_crps.reshape(B, 6),
        mean_rel,
        tstat,
        loss.log1p().reshape(B, 6),
        bias.asinh().reshape(B, 6),
        (cov - 0.8).reshape(B, 6),
        ok.reshape(B, 6),
    ], dim=-1).clamp(-8.0, 8.0)  # (B, 48)
    bt_f = bt_vec[:, None, :].expand(B, H, -1)
    return torch.cat([exp_f, dis, h_f, bt_f], dim=-1)  # 35 + 48 = 83


class UnclampedMetaCombiner(nn.Module):
    def __init__(self, mode="zero_replay", d_hid=96, depth=3):
        super().__init__()
        self.mode = mode
        self.use_backtest = (mode != "zero_replay")
        self.depth = depth if mode == "loop3" else 1
        in_dim = 83 if self.use_backtest else 35
        self.in_proj = nn.Sequential(nn.Linear(in_dim, d_hid), nn.LayerNorm(d_hid), nn.SiLU())
        if mode == "loop3":
            self.core = nn.Sequential(
                nn.LayerNorm(d_hid), nn.Linear(d_hid, d_hid * 2), nn.SiLU(),
                nn.Dropout(0.05), nn.Linear(d_hid * 2, d_hid),
            )
            self.step_emb = nn.Parameter(torch.zeros(6, 1, 1, d_hid))
            self.infi_q = nn.Parameter(torch.zeros(d_hid))
            self.infi_log_decay = nn.Parameter(torch.tensor(-1.5))
            self.infi_norm = nn.LayerNorm(d_hid)
        else:
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.LayerNorm(d_hid), nn.Linear(d_hid, d_hid * 2), nn.SiLU(),
                    nn.Dropout(0.05), nn.Linear(d_hid * 2, d_hid),
                ) for _ in range(2)
            ])
        self.out_norm = nn.LayerNorm(d_hid)
        self.head = nn.Linear(d_hid, 8)

    def forward(self, q_norm, feats, bt=None, override_depth=None):
        x_in = build_meta_features(q_norm, feats, bt, use_backtest=self.use_backtest)
        h0 = self.in_proj(x_in)
        if self.mode == "loop3":
            R = self.depth if override_depth is None else override_depth
            lam = F.softplus(self.infi_log_decay)
            s0 = (self.infi_norm(h0) * self.infi_q).sum(-1, keepdim=True)
            m_run = s0; num = h0; den = torch.ones_like(s0)
            h = h0
            scale_d = math.sqrt(max(1, R))
            for t in range(R):
                h = h + self.core(h + 0.2 / (t + 1.0) * (h0 - h) + self.step_emb[min(t, 5)]) / scale_d
                st = (self.infi_norm(h) * self.infi_q).sum(-1, keepdim=True)
                m_new = torch.maximum(m_run - lam, st)
                ap = torch.exp(m_run - lam - m_new)
                ac = torch.exp(st - m_new)
                num = ap * num + ac * h
                den = ap * den + ac
                m_run = m_new
            h = num / den.clamp_min(1e-6)
        else:
            h = h0
            for blk in self.blocks:
                h = h + blk(h)
        out = self.head(self.out_norm(h))
        w_med = F.softmax(out[..., 0:3], dim=-1)
        w_spr = F.softmax(out[..., 3:6], dim=-1)
        spr_scale = torch.exp(0.20 * torch.tanh(out[..., 6:7]))
        med_mix = 0.5 * torch.sigmoid(out[..., 7:8])
        qp = q_norm.permute(0, 2, 1, 3)
        meds = qp[..., 4:5]
        sprs = qp - meds
        q_lin = (w_med[..., None] * meds).sum(dim=2) + spr_scale * (w_spr[..., None] * sprs).sum(dim=2)
        q_rob = qp.median(dim=2).values
        q_out = (1.0 - med_mix) * q_lin + med_mix * q_rob
        q_out = torch.sort(q_out, dim=-1).values
        return q_out, w_med, w_spr


def _series_features_batch(x_np):
    """Exact 14 scale-invariant history stats matching build_multi_cache.py. x_np: (N, T) float32."""
    N, T = x_np.shape
    feats = np.zeros((N, 14), dtype=np.float32)
    locs = np.zeros(N, dtype=np.float32)
    scales = np.ones(N, dtype=np.float32)
    for i in range(N):
        r = x_np[i].astype(np.float64)
        v = np.isfinite(r)
        n = int(v.sum())
        if n < 2:
            if n == 1:
                locs[i] = float(r[v][0])
            continue
        rv = r[v]
        mu = float(rv.mean())
        sd = max(float(rv.std()), 1e-6)
        locs[i], scales[i] = mu, sd
        z = (r - mu) / sd
        zv = z[v]
        feats[i, 0] = math.log1p(n) / 10.0
        feats[i, 1] = float(v.mean())
        feats[i, 2] = float((rv == 0).mean())
        feats[i, 3] = float((rv >= 0).all())
        feats[i, 4] = float((rv >= 0).all() and np.allclose(rv, np.round(rv), atol=1e-5))
        feats[i, 5] = float(np.abs(zv).mean())
        dz = np.diff(zv)
        feats[i, 6] = float(np.abs(dz).mean()) if len(dz) else 0.0
        feats[i, 7] = float(zv[-min(32, len(zv)):].mean())
        feats[i, 8] = float(zv[-min(32, len(zv)):].std())
        feats[i, 9] = math.log1p(float(np.abs(zv).max()))
        feats[i, 10] = float(np.mean(zv[-min(16, len(zv)):]) - np.mean(zv[:min(16, len(zv))]))
        for k, lag in enumerate((1, 7, 24)):
            if len(zv) > lag + 2:
                feats[i, 11 + k] = float(np.mean(zv[lag:] * zv[:-lag]))
    return np.clip(np.nan_to_num(feats), -8.0, 8.0), locs, scales


def _cutoff_stats(q, y_t, sc_t, q_t):
    valid = torch.isfinite(y_t)
    diff = (y_t.nan_to_num(0.0)[..., None] - q) / sc_t[:, None, None]
    mae = (diff[..., 4].abs() * valid.float()).sum(-1) / valid.float().sum(-1).clamp_min(1.0)
    crps = (2.0 * torch.maximum(q_t * diff, (q_t - 1.0) * diff).mean(-1) * valid.float()).sum(-1) / valid.float().sum(-1).clamp_min(1.0)
    bias = (diff[..., 4] * valid.float()).sum(-1) / valid.float().sum(-1).clamp_min(1.0)
    cov = (((y_t >= q[..., 0]) & (y_t <= q[..., 8])) & valid).float().sum(-1) / valid.float().sum(-1).clamp_min(1.0)
    ok = (valid.sum(-1) >= 4).float()
    return torch.stack([mae, crps, bias, cov, ok], dim=-1)


class ApexForecaster:
    def __init__(self, native: TimestralForecaster, t0: T0Forecaster, chronos: Chronos2Pipeline,
                 meta: UnclampedMetaCombiner, policy: dict | None = None):
        self.native = native
        self.t0 = t0
        self.chronos = chronos
        self.meta = meta
        self.policy = policy or {"replay": False, "depth": 3, "min_replay_len": 160, "integer_snap": True, "nonneg_clip": True}
        self.max_context = native.max_context
        self.quantile_levels = native.quantile_levels
        levels = list(self.quantile_levels)
        self._decile_idx = [next(i for i, lv in enumerate(levels) if abs(lv - q) < 1e-9) for q in DECILES]
        self._q_t = torch.tensor(DECILES, dtype=torch.float32)

    @classmethod
    def from_pretrained(cls, path, device="cuda"):
        p = Path(path)
        native_cfg = TimestralConfig.load(p / "config.json")
        native_model = TimestralModel(native_cfg)
        t0_model = T0Forecaster(**json.loads((p / "t0_config.json").read_text()))
        ch_cfg = Chronos2CoreConfig(**json.loads((p / "chronos_config.json").read_text()))
        ch_model = Chronos2Model(ch_cfg)
        meta_cfg = json.loads((p / "meta_config.json").read_text())
        meta = UnclampedMetaCombiner(**meta_cfg)
        policy = json.loads((p / "policy.json").read_text())

        weights = load_file(str(p / "model.safetensors"))
        native_model.load_state_dict({k[7:]: v for k, v in weights.items() if k.startswith("native.")}, strict=True)
        t0_model.load_state_dict({k[3:]: v for k, v in weights.items() if k.startswith("t0.")}, strict=True)
        ch_model.load_state_dict({k[8:]: v for k, v in weights.items() if k.startswith("chronos.")}, strict=True)
        meta.load_state_dict({k[5:]: v for k, v in weights.items() if k.startswith("meta.")}, strict=True)
        del weights

        native_model = native_model.to(device).eval()
        native_model.transformer.rope.rebuild_inv_freq()
        t0_model = t0_model.to(device).eval()
        ch_model = ch_model.to(device=device, dtype=torch.bfloat16).eval()
        meta = meta.to(device).eval()
        return cls(TimestralForecaster(native_model), t0_model, Chronos2Pipeline(ch_model), meta, policy=policy)

    @torch.inference_mode()
    def _run_t0(self, x, h, past_cov, known_cov, fallback):
        """Run T0-gamma with official a2 settings, covariate support, and safe chunking/fallback."""
        B, V, T = x.shape
        device = x.device
        t_sl = min(T, 8192)
        t0_start = T - t_sl
        x8 = x[..., t0_start:]
        p8 = past_cov[..., t0_start:] if past_cov is not None else None
        k8 = known_cov[..., t0_start:T + h] if known_cov is not None else None
        ctx = torch.cat([x8, p8], dim=1) if p8 is not None else x8

        def _exec_t0(ctx_in, k_in, n_tgt):
            blocks = []
            for i in range(B):
                row = ctx_in[i]
                valid = row.isfinite().any(0)
                if k_in is not None:
                    valid = valid | k_in[i, :, :t_sl].isfinite().any(0)
                ii = torch.where(valid)[0]
                s = int(ii[0]) if len(ii) else t_sl - 1
                if k_in is None:
                    blocks.append(row[:, s:])
                else:
                    blocks.append(TimeSeries.from_array(row[None, :, s:], k_in[i:i+1, :, s:]))
            if k_in is None:
                arr, mask, ids = batch_series(blocks)
                out = self.t0.predict(
                    arr.to(device), horizon=h, mask=mask.to(device), group_ids=ids.to(device),
                    quantile_levels=DECILES, trim_leading_missing=True, symmetric_averaging=True,
                    clip_nonnegative=True, snap_integers=True,
                ).quantiles.float()
            else:
                batch = TimeSeries.batch(blocks)
                out = self.t0.predict(
                    batch, horizon=h, quantile_levels=DECILES,
                    trim_leading_missing=True, symmetric_averaging=False,
                    clip_nonnegative=True, snap_integers=True,
                ).quantiles.float()
            return out.reshape(B, ctx_in.shape[1], h, 9)[:, :n_tgt]

        try:
            if ctx.shape[1] <= 24:
                q = _exec_t0(ctx, k8, V)
            else:
                # Chunk variates cleanly in groups of 16 so large-V multivariate datasets never OOM
                parts = []
                for v0 in range(0, V, 16):
                    v1 = min(V, v0 + 16)
                    c_sub = torch.cat([x8[:, v0:v1], p8], dim=1) if p8 is not None else x8[:, v0:v1]
                    parts.append(_exec_t0(c_sub, k8, v1 - v0))
                q = torch.cat(parts, dim=1)
            bad = ~torch.isfinite(q).all(dim=(-2, -1))
            if bad.any():
                q = torch.where(bad[..., None, None], fallback, q)
            return q
        except Exception:
            torch.cuda.empty_cache()
            return fallback.clone()

    @torch.inference_mode()
    def _run_chronos(self, x, h, past_cov, known_cov, fallback):
        """Run Chronos-2 with a2 sym-avg (no covs) or native covariate dict support, singleton-invariant."""
        B, V, T = x.shape
        device = x.device
        t_sl = min(T, 8192)
        t0_start = T - t_sl
        x8 = x[..., t0_start:].detach().cpu().float()
        p8 = past_cov[..., t0_start:].detach().cpu().float() if past_cov is not None else None
        k8 = known_cov[..., t0_start:T + h].detach().cpu().float() if known_cov is not None else None
        has_cov = (p8 is not None) or (k8 is not None)

        try:
            if not has_cov:
                rows = []
                for i in range(B):
                    row = x8[i]  # (V, t_sl)
                    valid = row.isfinite().any(0)
                    ii = torch.where(valid)[0]
                    s = int(ii[0]) if len(ii) else t_sl - 1
                    rows.append(row[:, s:])
                p1, _ = self.chronos.predict_quantiles(rows, prediction_length=h, quantile_levels=DECILES, cross_learning=False)
                p2, _ = self.chronos.predict_quantiles([-r for r in rows], prediction_length=h, quantile_levels=DECILES, cross_learning=False)
                q = 0.5 * (torch.stack(p1).float() + (-torch.stack(p2).float()).flip(-1))
            else:
                inps = []
                for i in range(B):
                    row = x8[i]
                    valid = row.isfinite().any(0)
                    ii = torch.where(valid)[0]
                    s = int(ii[0]) if len(ii) else t_sl - 1
                    d = {"target": row[:, s:].numpy()}
                    pc, fc = {}, {}
                    if p8 is not None:
                        for c in range(p8.shape[1]):
                            arr = p8[i, c, s:].numpy()
                            if np.isfinite(arr).any():
                                pc[f"p{c}"] = arr
                    if k8 is not None:
                        for c in range(k8.shape[1]):
                            arr_p = k8[i, c, s:t_sl].numpy()
                            arr_f = k8[i, c, t_sl:t_sl + h].numpy()
                            if np.isfinite(arr_p).any() and np.isfinite(arr_f).any():
                                pc[f"k{c}"] = arr_p
                                fc[f"k{c}"] = arr_f
                    if pc:
                        d["past_covariates"] = pc
                    if fc:
                        d["future_covariates"] = fc
                    inps.append(d)
                p1, _ = self.chronos.predict_quantiles(inps, prediction_length=h, quantile_levels=DECILES, cross_learning=False)
                q = torch.stack(p1).float()

            q = q.to(device)
            # Nonnegative clamp per (b, v) where observed history >= 0
            obs = torch.isfinite(x)
            all_nonneg = ((~obs) | (x >= 0)).all(dim=-1) & obs.any(dim=-1)
            q = torch.where(all_nonneg[..., None, None], q.clamp_min(0.0), q)
            bad = ~torch.isfinite(q).all(dim=(-2, -1))
            if bad.any():
                q = torch.where(bad[..., None, None], fallback, q)
            return q
        except Exception:
            torch.cuda.empty_cache()
            return fallback.clone()

    def _extract_covs(self, bs, t, h, x_ref, kwargs):
        pasts, knowns = [], []
        for name in ("past_covariates", "categorical_past_covariates", "known_covariates", "categorical_known_covariates"):
            val = kwargs.get(name)
            if val is None:
                continue
            known = "known" in name
            cov = self.native._as_covariates(val, bs, t + h if known else t, name).to(x_ref)
            if known:
                knowns.append(cov)
            else:
                pasts.append(cov)
        return (torch.cat(pasts, 1) if pasts else None), (torch.cat(knowns, 1) if knowns else None)

    def _run_all_3(self, x, h, kwargs):
        nat_out = self.native.forecast(x, h, **kwargs)
        q_tm_21 = nat_out.quantiles.float()
        q_tm = q_tm_21[..., self._decile_idx]
        past_cov, known_cov = self._extract_covs(x.shape[0], x.shape[2], h, x, kwargs)
        q_t0 = self._run_t0(x, h, past_cov, known_cov, q_tm)
        q_ch = self._run_chronos(x, h, past_cov, known_cov, q_tm)
        return nat_out, q_tm_21, q_tm, q_t0, q_ch

    def _embed_21(self, q9, q_tm_21):
        """Embed exact 9 deciles [0.1..0.9] into Timestral's 21-level grid smoothly and monotonically."""
        out = q_tm_21.clone()
        idx = self._decile_idx  # [2, 4, 6, 8, 10, 12, 14, 16, 18]
        for k, pos in enumerate(idx):
            out[..., pos] = q9[..., k]
        # Linearly interpolate interior half-deciles [0.15, 0.25, ..., 0.85]
        for k in range(8):
            lo_p, hi_p = idx[k], idx[k + 1]
            if hi_p - lo_p == 2:
                out[..., lo_p + 1] = 0.5 * (q9[..., k] + q9[..., k + 1])
        # Anchor extreme outer tails [0.01, 0.05] and [0.95, 0.99] via native tail offsets from q10/q90
        if idx[0] >= 2:
            out[..., 1] = q9[..., 0] + (q_tm_21[..., 1] - q_tm_21[..., idx[0]])
            out[..., 0] = q9[..., 0] + (q_tm_21[..., 0] - q_tm_21[..., idx[0]])
        if idx[-1] <= out.shape[-1] - 3:
            out[..., -2] = q9[..., -1] + (q_tm_21[..., -2] - q_tm_21[..., idx[-1]])
            out[..., -1] = q9[..., -1] + (q_tm_21[..., -1] - q_tm_21[..., idx[-1]])
        return torch.sort(out, dim=-1).values

    @torch.inference_mode()
    def forecast(self, context, prediction_length, observation_mask=None, **kwargs):
        x, seen = self.native._as_context(context, observation_mask)
        x = x.masked_fill(~seen, float("nan")).float()
        B, V, T = x.shape
        H = int(prediction_length)
        device = x.device
        N = B * V

        nat_out, q_tm_21, q_tm, q_t0, q_ch = self._run_all_3(x, H, kwargs)
        x_flat_np = x.reshape(N, T).detach().cpu().numpy()
        feats_np, locs_np, scales_np = _series_features_batch(x_flat_np)
        feats_t = torch.from_numpy(feats_np).to(device)
        loc_t = torch.from_numpy(locs_np).to(device)
        sc_t = torch.from_numpy(scales_np).to(device)

        q_stack = torch.stack([q_tm.reshape(N, H, 9), q_t0.reshape(N, H, 9), q_ch.reshape(N, H, 9)], dim=1)
        q_norm = ((q_stack - loc_t[:, None, None, None]) / sc_t[:, None, None, None]).clamp(-1e3, 1e3)

        bt_t = None
        if self.meta.use_backtest:
            bt_t = torch.zeros((N, 2, 3, 5), device=device, dtype=torch.float32)
            min_len = int(self.policy.get("min_replay_len", 160))
            w_bt = min(64, H, T // 6)
            if self.policy.get("replay", False) and T >= min_len and w_bt >= 4 and T >= 2 * w_bt + 32:
                q_t_dev = self._q_t.to(device)
                for c in range(2):
                    cut = T - (2 - c) * w_bt
                    kw_c = {}
                    for name in ("past_covariates", "categorical_past_covariates", "known_covariates", "categorical_known_covariates"):
                        val = kwargs.get(name)
                        if val is None:
                            continue
                        arr = self.native._as_covariates(val, B, T + H if "known" in name else T, name).to(x)
                        kw_c[name] = arr[..., :cut + w_bt] if "known" in name else arr[..., :cut]
                    _, _, b_tm, b_t0, b_ch = self._run_all_3(x[..., :cut], w_bt, kw_c)
                    y_bt = x[..., cut:cut + w_bt].reshape(N, w_bt)
                    bt_t[:, c, 0] = _cutoff_stats(b_tm.reshape(N, w_bt, 9), y_bt, sc_t, q_t_dev)
                    bt_t[:, c, 1] = _cutoff_stats(b_t0.reshape(N, w_bt, 9), y_bt, sc_t, q_t_dev)
                    bt_t[:, c, 2] = _cutoff_stats(b_ch.reshape(N, w_bt, 9), y_bt, sc_t, q_t_dev)

        depth = self.policy.get("depth", None)
        q_comb_norm, _, _ = self.meta(q_norm, feats_t, bt_t, override_depth=depth)
        q_comb = (q_comb_norm * sc_t[:, None, None] + loc_t[:, None, None]).reshape(B, V, H, 9)

        # Domain-exact post-processing
        if self.policy.get("nonneg_clip", True):
            obs = torch.isfinite(x)
            all_nonneg = ((~obs) | (x >= 0)).all(dim=-1) & obs.any(dim=-1)
            q_comb = torch.where(all_nonneg[..., None, None], q_comb.clamp_min(0.0), q_comb)
        if self.policy.get("integer_snap", True) and T >= 84:
            tail = x[..., -84:]
            elig = (torch.isfinite(tail) & (tail >= 0) & (tail == tail.round())).all(dim=-1)
            if elig.any():
                q_comb = torch.where(elig[..., None, None], q_comb.round(), q_comb)
        q_comb = torch.sort(q_comb, dim=-1).values

        q_21 = self._embed_21(q_comb, q_tm_21)
        return type(nat_out)(quantiles=q_21.to(nat_out.quantiles.dtype), quantile_levels=self.quantile_levels)
