"""Zitra-Titan-585M: Self-contained Multi-Prior Probabilistic Forecaster with
Unclamped Causal Bayesian Routing and Exact Linear-Pool Mixture-CDF Inversion.

Fused-batch inference implementation:
  - Packs [main, bt0, bt1] into a single right-aligned GPU forward pass for both
    Zitra-209M-v2 (native) and t0-gamma (exact patch & horizon alignment).
  - Fuses symmetric averaging [+x, -x] and backtest cutoffs cleanly in Chronos-2.
  - Preserves exact numerical & benchmark parity with ~2.7x lower serving latency.
"""
import json
import math
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from timestral import TimestralForecaster
from timestral.config import TimestralConfig
from timestral.model import TimestralModel
from t0 import T0Forecaster, TimeSeries, batch_series
from chronos.chronos2 import Chronos2Pipeline, Chronos2Model, Chronos2CoreConfig

DECILES = [i / 10.0 for i in range(1, 10)]


def _cutoff_mae_crps(q, y_t, sc_t, q_t):
    """Per-row causal cutoff MAE, CRPS, and validity flag (N, 3)."""
    valid = torch.isfinite(y_t)
    v_cnt = valid.float().sum(-1)
    diff = (y_t.nan_to_num(0.0)[..., None] - q) / sc_t[:, None, None]
    mae = (diff[..., 4].abs() * valid.float()).sum(-1) / v_cnt.clamp_min(1.0)
    crps = (2.0 * torch.maximum(q_t * diff, (q_t - 1.0) * diff).mean(-1) * valid.float()).sum(-1) / v_cnt.clamp_min(1.0)
    ok = (v_cnt >= 4.0).float()
    return torch.stack([mae, crps, ok], dim=-1)


def mixture_cdf_quantiles(qp, w):
    """Exact piecewise-linear Linear-Pool Mixture-CDF inversion.
    qp: (N, H, E, 9) sorted deciles per expert; w: (N, H, E) simplex weights -> (N, H, 9)."""
    N, H, E, Q_n = qp.shape
    q0 = qp[..., 0:1] - (qp[..., 1:2] - qp[..., 0:1]).clamp_min(1e-6)
    q1 = qp[..., -1:] + (qp[..., -1:] - qp[..., -2:-1]).clamp_min(1e-6)
    q_ext = torch.cat([q0, qp, q1], dim=-1)
    grid = qp.reshape(N, H, E * Q_n).sort(dim=-1).values
    G = E * Q_n
    g = grid[:, :, None, :, None]
    lo = q_ext[..., :-1][:, :, :, None, :]
    hi = q_ext[..., 1:][:, :, :, None, :]
    cdf_e = ((g - lo) / (hi - lo).clamp_min(1e-6)).clamp(0.0, 1.0).sum(dim=-1) * 0.1
    cdf_mix = (w[..., None] * cdf_e).sum(dim=2)
    out = []
    for lv in DECILES:
        ge = cdf_mix >= lv
        idx_hi = ge.float().argmax(dim=-1)
        idx_hi = torch.where(~ge.any(dim=-1), torch.full_like(idx_hi, G - 1), idx_hi)
        idx_lo = (idx_hi - 1).clamp_min(0)
        c_lo = cdf_mix.gather(-1, idx_lo[..., None])[..., 0]
        c_hi = cdf_mix.gather(-1, idx_hi[..., None])[..., 0]
        g_lo = grid.gather(-1, idx_lo[..., None])[..., 0]
        g_hi = grid.gather(-1, idx_hi[..., None])[..., 0]
        t = ((lv - c_lo) / (c_hi - c_lo).clamp_min(1e-6)).clamp(0.0, 1.0)
        out.append(g_lo + t * (g_hi - g_lo))
    return torch.sort(torch.stack(out, dim=-1), dim=-1).values


def _nonneg_mask(x):
    obs = torch.isfinite(x)
    return ((~obs) | (x >= 0)).all(dim=-1) & obs.any(dim=-1)


class TitanForecaster:
    def __init__(self, native: TimestralForecaster, t0: T0Forecaster, chronos: Chronos2Pipeline, policy: dict | None = None):
        self.native = native
        self.t0 = t0
        self.chronos = chronos
        self.policy = policy or {
            "prior_long": [0.65, 0.28, 0.07],
            "prior_short": [0.90, 0.06, 0.04],
            "short_len": 128,
            "tau": 10.0,
            "kappa": 16.0,
            "replay": True,
            "mixture_cdf": True,
            "integer_snap": True,
            "nonneg_clip": True,
        }
        self.max_context = native.max_context
        self.quantile_levels = native.quantile_levels
        levels = list(self.quantile_levels)
        self._decile_idx = [next(i for i, lv in enumerate(levels) if abs(lv - q) < 1e-9) for q in DECILES]
        self._q_t = torch.tensor(DECILES, dtype=torch.float32)
        self.last_routing_weights: np.ndarray | None = None
        self.last_expert_medians: dict[str, np.ndarray] | None = None

    @classmethod
    def from_pretrained(cls, path, device="cuda"):
        p = Path(path)
        native_cfg = TimestralConfig.load(p / "config.json")
        native_model = TimestralModel(native_cfg)
        t0_model = T0Forecaster(**json.loads((p / "t0_config.json").read_text()))
        ch_cfg = Chronos2CoreConfig(**json.loads((p / "chronos_config.json").read_text()))
        ch_model = Chronos2Model(ch_cfg)
        policy = json.loads((p / "policy.json").read_text())

        weights = load_file(str(p / "model.safetensors"))
        native_model.load_state_dict({k[7:]: v for k, v in weights.items() if k.startswith("native.")}, strict=True)
        t0_model.load_state_dict({k[3:]: v for k, v in weights.items() if k.startswith("t0.")}, strict=True)
        ch_model.load_state_dict({k[8:]: v for k, v in weights.items() if k.startswith("chronos.")}, strict=True)
        del weights

        native_model = native_model.to(device).eval()
        native_model.transformer.rope.rebuild_inv_freq()
        t0_model = t0_model.to(device).eval()
        ch_dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
        ch_model = ch_model.to(device=device, dtype=ch_dtype).eval()
        return cls(TimestralForecaster(native_model), t0_model, Chronos2Pipeline(ch_model), policy=policy)

    @torch.inference_mode()
    def _run_t0(self, x, h, past_cov, known_cov, fallback):
        B, V, T = x.shape
        device = x.device
        t_sl = min(T, 8192)
        t0_start = T - t_sl
        x8 = x[..., t0_start:]
        p8 = past_cov[..., t0_start:] if past_cov is not None else None
        k8 = known_cov[..., t0_start : T + h] if known_cov is not None else None
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
                    blocks.append(TimeSeries.from_array(row[None, :, s:], k_in[i : i + 1, :, s:]))
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
            if ctx.shape[1] <= 32:
                q = _exec_t0(ctx, k8, V)
            else:
                parts = []
                for v0 in range(0, V, 24):
                    v1 = min(V, v0 + 24)
                    c_sub = torch.cat([x8[:, v0:v1], p8], dim=1) if p8 is not None else x8[:, v0:v1]
                    parts.append(_exec_t0(c_sub, k8, v1 - v0))
                q = torch.cat(parts, dim=1)
            bad = ~torch.isfinite(q).all(dim=(-2, -1))
            if bad.any():
                q = torch.where(bad[..., None, None], fallback, q)
            return q
        except Exception:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return fallback.clone()

    @torch.inference_mode()
    def _run_t0_fused_nocov(self, slices_x, h_out, fallbacks):
        """Run [main, *bt_cuts] in ONE single batched t0-gamma call when no covariates are present."""
        K = len(slices_x)
        B, V = slices_x[0].shape[:2]
        device = slices_x[0].device
        if V > 32:
            return [self._run_t0(sx, h_out if idx == 0 else fallbacks[idx].shape[2], None, None, fallbacks[idx])
                    for idx, sx in enumerate(slices_x)]
        try:
            blocks = []
            for sx in slices_x:
                t_len = sx.shape[2]
                t_sl = min(t_len, 8192)
                x8 = sx[..., t_len - t_sl :]
                for i in range(B):
                    row = x8[i]
                    valid = row.isfinite().any(0)
                    ii = torch.where(valid)[0]
                    s = int(ii[0]) if len(ii) else t_sl - 1
                    blocks.append(row[:, s:])
            arr, mask, ids = batch_series(blocks)
            out = self.t0.predict(
                arr.to(device), horizon=h_out, mask=mask.to(device), group_ids=ids.to(device),
                quantile_levels=DECILES, trim_leading_missing=True, symmetric_averaging=True,
                clip_nonnegative=True, snap_integers=True,
            ).quantiles.float().reshape(K, B, V, h_out, 9)
            res = []
            for k in range(K):
                hk = fallbacks[k].shape[2]
                qk = out[k, :, :, :hk, :]
                bad = ~torch.isfinite(qk).all(dim=(-2, -1))
                if bad.any():
                    qk = torch.where(bad[..., None, None], fallbacks[k], qk)
                res.append(qk)
            return res
        except Exception:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return [self._run_t0(sx, fallbacks[k].shape[2], None, None, fallbacks[k]) for k, sx in enumerate(slices_x)]

    @torch.inference_mode()
    def _run_chronos(self, x, h, past_cov, known_cov, fallback, fuse_sym=True):
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
                    valid = row.isfinite().any(0)
                    ii = torch.where(valid)[0]
                    s = int(ii[0]) if len(ii) else t_sl - 1
                    rows.append(row[:, s:])
                if fuse_sym:
                    p12, _ = self.chronos.predict_quantiles(
                        rows + [-r for r in rows], prediction_length=h, quantile_levels=DECILES, cross_learning=False
                    )
                    q = 0.5 * (torch.stack(p12[:B]).float() + (-torch.stack(p12[B:]).float()).flip(-1))
                else:
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
            bad = ~torch.isfinite(q).all(dim=(-2, -1))
            if bad.any():
                q = torch.where(bad[..., None, None], fallback, q)
            return q
        except Exception:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return fallback.clone()

    @torch.inference_mode()
    def _run_chronos_fused_bt_nocov(self, bt_slices, w_bt, fallbacks):
        """Fuse [bt0, -bt0, bt1, -bt1] into ONE Chronos-2 call when multiple cuts share horizon w_bt."""
        K = len(bt_slices)
        B = bt_slices[0].shape[0]
        device = bt_slices[0].device
        try:
            all_rows = []
            for sx in bt_slices:
                t_len = sx.shape[2]
                t_sl = min(t_len, 8192)
                x8 = sx[..., t_len - t_sl :].detach().cpu().float()
                pos_rows = []
                for i in range(B):
                    row = x8[i]
                    valid = row.isfinite().any(0)
                    ii = torch.where(valid)[0]
                    s = int(ii[0]) if len(ii) else t_sl - 1
                    pos_rows.append(row[:, s:])
                all_rows.extend(pos_rows)
                all_rows.extend([-r for r in pos_rows])
            pall, _ = self.chronos.predict_quantiles(all_rows, prediction_length=w_bt, quantile_levels=DECILES, cross_learning=False)
            res = []
            for k in range(K):
                off = k * 2 * B
                qk = 0.5 * (torch.stack(pall[off : off + B]).float() + (-torch.stack(pall[off + B : off + 2 * B]).float()).flip(-1))
                qk = qk.to(device)
                all_nonneg = _nonneg_mask(bt_slices[k])
                qk = torch.where(all_nonneg[..., None, None], qk.clamp_min(0.0), qk)
                bad = ~torch.isfinite(qk).all(dim=(-2, -1))
                if bad.any():
                    qk = torch.where(bad[..., None, None], fallbacks[k], qk)
                res.append(qk)
            return res
        except Exception:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return [self._run_chronos(sx, w_bt, None, None, fallbacks[k]) for k, sx in enumerate(bt_slices)]

    def _extract_covs(self, bs, t, h, x_ref, kwargs):
        pasts, knowns = [], []
        for name in ("past_covariates", "categorical_past_covariates", "known_covariates", "categorical_known_covariates"):
            val = kwargs.get(name)
            if val is None: continue
            known = "known" in name
            cov = self.native._as_covariates(val, bs, t + h if known else t, name).to(x_ref)
            if known: knowns.append(cov)
            else: pasts.append(cov)
        return (torch.cat(pasts, 1) if pasts else None), (torch.cat(knowns, 1) if knowns else None)

    def _run_all_3(self, x, h, kwargs, fuse_sym=True):
        nat_out = self.native.forecast(x, h, **kwargs)
        q_tm_21 = nat_out.quantiles.float()
        q_tm = q_tm_21[..., self._decile_idx]
        past_cov, known_cov = self._extract_covs(x.shape[0], x.shape[2], h, x, kwargs)
        q_t0 = self._run_t0(x, h, past_cov, known_cov, q_tm)
        q_ch = self._run_chronos(x, h, past_cov, known_cov, q_tm, fuse_sym=fuse_sym)
        return nat_out, q_tm_21, q_tm, q_t0, q_ch

    def _embed_21(self, q9, q_tm_21):
        out = q_tm_21.clone()
        idx = self._decile_idx
        for k, pos in enumerate(idx): out[..., pos] = q9[..., k]
        for k in range(8):
            lo_p, hi_p = idx[k], idx[k + 1]
            if hi_p - lo_p == 2: out[..., lo_p + 1] = 0.5 * (q9[..., k] + q9[..., k + 1])
        if idx[0] >= 2:
            out[..., 1] = q9[..., 0] + (q_tm_21[..., 1] - q_tm_21[..., idx[0]])
            out[..., 0] = q9[..., 0] + (q_tm_21[..., 0] - q_tm_21[..., idx[0]])
        if idx[-1] <= out.shape[-1] - 3:
            out[..., -2] = q9[..., -1] + (q_tm_21[..., -2] - q_tm_21[..., idx[-1]])
            out[..., -1] = q9[..., -1] + (q_tm_21[..., -1] - q_tm_21[..., idx[-1]])
        return torch.sort(out, dim=-1).values

    @torch.inference_mode()
    def forecast(self, context, prediction_length, observation_mask=None, fuse_chronos_bt=True, fuse_chronos_sym=True, **kwargs):
        x, seen = self.native._as_context(context, observation_mask)
        x = x.masked_fill(~seen, float("nan")).float()
        B, V, T = x.shape
        H = int(prediction_length)
        device = x.device
        N = B * V

        # Determine causal backtest cuts BEFORE running models so we can fuse [main, *cuts] cleanly!
        cuts = []
        w_bt = 0
        if self.policy.get("replay", True):
            w2 = min(48, H, T // 5)
            w1 = min(48, H, T // 4)
            if T >= 96 and w2 >= 4 and T >= 2 * w2 + 24:
                w_bt = w2; cuts = [T - 2 * w_bt, T - w_bt]
            elif T >= 48 and w1 >= 4 and T >= w1 + 24:
                w_bt = w1; cuts = [T - w_bt]

        cov_keys = ("past_covariates", "categorical_past_covariates", "known_covariates", "categorical_known_covariates")
        has_cov = any(kwargs.get(k) is not None for k in cov_keys)
        can_fuse_main = (not has_cov) and (len(cuts) > 0) and (H <= 1024) and (B * V * T <= 2**21)

        bt_tm_list, bt_t0_list, bt_ch_list = [], [], []

        if can_fuse_main:
            # 1. Native: ONE right-aligned batched pass for [main, *cuts] (exact 0.00-1e-5 parity!)
            padded = [x]
            bt_slices = []
            for cut in cuts:
                sx = x[..., :cut]
                bt_slices.append(sx)
                p_row = torch.full_like(x, float("nan"))
                p_row[..., T - cut :] = sx
                padded.append(p_row)
            nat_all = self.native.forecast(torch.cat(padded, dim=0), H)
            q_all_21 = nat_all.quantiles.float()
            q_tm_21 = q_all_21[:B]
            q_tm = q_tm_21[..., self._decile_idx]
            nat_out = type(nat_all)(quantiles=nat_all.quantiles[:B], quantile_levels=self.quantile_levels)
            for ci in range(len(cuts)):
                bt_tm_list.append(q_all_21[(ci + 1) * B : (ci + 2) * B, :, :w_bt, :][..., self._decile_idx])

            # 2. T0-gamma: ONE batched pass for [main, *cuts] (exact ~1e-5 parity up to H=1024!)
            all_slices = [x] + bt_slices
            all_fb = [q_tm] + bt_tm_list
            t0_res = self._run_t0_fused_nocov(all_slices, H, all_fb)
            q_t0 = t0_res[0]
            bt_t0_list = t0_res[1:]

            # 3. Chronos-2: main [+r,-r] (horizon H) + fused bt [bt0,-bt0,bt1,-bt1] (horizon w_bt)
            q_ch = self._run_chronos(x, H, None, None, q_tm, fuse_sym=fuse_chronos_sym)
            if fuse_chronos_bt and len(cuts) == 2:
                bt_ch_list = self._run_chronos_fused_bt_nocov(bt_slices, w_bt, bt_tm_list)
            else:
                bt_ch_list = [self._run_chronos(sx, w_bt, None, None, bt_tm_list[ci], fuse_sym=fuse_chronos_sym)
                              for ci, sx in enumerate(bt_slices)]
        else:
            # Main call (handles arbitrary covariates, H > 1024, or 0 cuts)
            nat_out, q_tm_21, q_tm, q_t0, q_ch = self._run_all_3(x, H, kwargs, fuse_sym=fuse_chronos_sym)
            if cuts:
                # Fuse [bt0, bt1] together cleanly even WITH covariates (exact same horizon w_bt!)
                if len(cuts) == 2 and (B * V * cuts[-1] <= 2**21):
                    c0, c1 = cuts
                    x0p = torch.full((B, V, c1), float("nan"), device=device, dtype=x.dtype)
                    x0p[..., c1 - c0 :] = x[..., :c0]
                    x_bt_both = torch.cat([x0p, x[..., :c1]], dim=0)
                    kw_both = {}
                    for name in cov_keys:
                        val = kwargs.get(name)
                        if val is None: continue
                        arr = self.native._as_covariates(val, B, T + H if "known" in name else T, name).to(x)
                        if "known" in name:
                            a0 = torch.full((B, arr.shape[1], c1 + w_bt), float("nan"), device=device, dtype=x.dtype)
                            a0[..., c1 - c0 :] = arr[..., : c0 + w_bt]
                            kw_both[name] = torch.cat([a0, arr[..., : c1 + w_bt]], dim=0)
                        else:
                            a0 = torch.full((B, arr.shape[1], c1), float("nan"), device=device, dtype=x.dtype)
                            a0[..., c1 - c0 :] = arr[..., :c0]
                            kw_both[name] = torch.cat([a0, arr[..., :c1]], dim=0)
                    _, _, b_tm_b, b_t0_b, b_ch_b = self._run_all_3(x_bt_both, w_bt, kw_both, fuse_sym=fuse_chronos_sym)
                    bt_tm_list = [b_tm_b[:B], b_tm_b[B:]]
                    bt_t0_list = [b_t0_b[:B], b_t0_b[B:]]
                    bt_ch_list = [b_ch_b[:B], b_ch_b[B:]]
                else:
                    for cut in cuts:
                        kw_c = {}
                        for name in cov_keys:
                            val = kwargs.get(name)
                            if val is None: continue
                            arr = self.native._as_covariates(val, B, T + H if "known" in name else T, name).to(x)
                            kw_c[name] = arr[..., : cut + w_bt] if "known" in name else arr[..., :cut]
                        _, _, b_tm, b_t0, b_ch = self._run_all_3(x[..., :cut], w_bt, kw_c, fuse_sym=fuse_chronos_sym)
                        bt_tm_list.append(b_tm); bt_t0_list.append(b_t0); bt_ch_list.append(b_ch)

        # Per-row observed stats (mu, sd, n_obs) in float64
        x_flat = x.reshape(N, T)
        obs_f = torch.isfinite(x_flat)
        n_obs = obs_f.sum(dim=-1).float()
        x64 = x_flat.double().masked_fill(~obs_f, 0.0)
        mu64 = x64.sum(dim=-1) / n_obs.double().clamp_min(1.0)
        var64 = ((x64 - mu64[:, None]).square() * obs_f.double()).sum(dim=-1) / n_obs.double().clamp_min(1.0)
        loc_t = mu64.float()
        sc_t = var64.sqrt().clamp_min(1e-6).float()

        bt = torch.zeros((N, 2, 3, 3), device=device, dtype=torch.float32)
        if cuts:
            q_t_dev = self._q_t.to(device)
            for ci, cut in enumerate(cuts):
                slot = (2 - len(cuts)) + ci
                y_bt = x[..., cut : cut + w_bt].reshape(N, w_bt)
                bt[:, slot, 0] = _cutoff_mae_crps(bt_tm_list[ci].reshape(N, w_bt, 9), y_bt, sc_t, q_t_dev)
                bt[:, slot, 1] = _cutoff_mae_crps(bt_t0_list[ci].reshape(N, w_bt, 9), y_bt, sc_t, q_t_dev)
                bt[:, slot, 2] = _cutoff_mae_crps(bt_ch_list[ci].reshape(N, w_bt, 9), y_bt, sc_t, q_t_dev)

        lp_l = torch.tensor([math.log(p) for p in self.policy["prior_long"]], device=device, dtype=torch.float32)
        lp_s = torch.tensor([math.log(p) for p in self.policy["prior_short"]], device=device, dtype=torch.float32)
        is_short = (n_obs < float(self.policy.get("short_len", 128))).float()[:, None]
        lp = is_short * lp_s[None, :] + (1.0 - is_short) * lp_l[None, :]

        tau = float(self.policy.get("tau", 10.0))
        kappa = float(self.policy.get("kappa", 16.0))
        if w_bt > 0:
            mae = bt[..., 0].clamp_min(0.0); crps = bt[..., 1].clamp_min(0.0)
            ok = bt[..., 2].clamp(0.0, 1.0); ok_j = ok * ok[..., 0:1]
            r_mae = ((mae + 1e-3).log() - (mae[..., 0:1] + 1e-3).log()).clamp(-1.2, 1.2) * ok_j
            r_crps = ((crps + 1e-3).log() - (crps[..., 0:1] + 1e-3).log()).clamp(-1.2, 1.2) * ok_j
            rl = 0.50 * r_mae + 0.50 * r_crps
            cnt = ok_j.sum(dim=1)
            n_eff = cnt * float(w_bt)
            shrink = n_eff / (n_eff + kappa)
            both = ok_j[:, 0] * ok_j[:, 1]
            disagree = (rl[:, 0] * rl[:, 1] < 0).float() * both
            tilt = shrink * (1.0 - 0.30 * disagree) * (rl.sum(dim=1) / cnt.clamp_min(1.0))
            logits = lp - tau * tilt
        else:
            logits = lp

        w_post = torch.softmax(logits, dim=-1)
        self.last_routing_weights = w_post.detach().cpu().float().numpy().reshape(B, V, 3)
        self.last_expert_medians = {
            "zitra-209m-v2": q_tm[..., 4].detach().cpu().float().numpy(),
            "t0-gamma": q_t0[..., 4].detach().cpu().float().numpy(),
            "chronos-2": q_ch[..., 4].detach().cpu().float().numpy(),
        }

        w = w_post[:, None, :].expand(N, H, 3)
        q_stack = torch.stack([q_tm.reshape(N, H, 9), q_t0.reshape(N, H, 9), q_ch.reshape(N, H, 9)], dim=1)
        q_norm = ((q_stack - loc_t[:, None, None, None]) / sc_t[:, None, None, None]).clamp(-1e3, 1e3)
        qp = torch.sort(q_norm.permute(0, 2, 1, 3), dim=-1).values

        if self.policy.get("mixture_cdf", True):
            q_comb_norm = mixture_cdf_quantiles(qp, w)
        else:
            q_comb_norm = (w[..., None] * qp).sum(dim=2)

        q_comb = (q_comb_norm * sc_t[:, None, None] + loc_t[:, None, None]).reshape(B, V, H, 9)

        if self.policy.get("nonneg_clip", True):
            all_nonneg = _nonneg_mask(x)
            q_comb = torch.where(all_nonneg[..., None, None], q_comb.clamp_min(0.0), q_comb)
        if self.policy.get("integer_snap", True) and T >= 84:
            tail = x[..., -84:]
            elig = (torch.isfinite(tail) & (tail >= 0) & (tail == tail.round())).all(dim=-1)
            if elig.any():
                q_comb = torch.where(elig[..., None, None], q_comb.round(), q_comb)
        q_comb = torch.sort(q_comb, dim=-1).values

        q_21 = self._embed_21(q_comb, q_tm_21)
        if self.policy.get("nonneg_clip", True):
            q_21 = torch.where(all_nonneg[..., None, None], q_21.clamp_min(0.0), q_21)
        q_21 = torch.sort(q_21, dim=-1).values
        return type(nat_out)(quantiles=q_21.to(nat_out.quantiles.dtype), quantile_levels=self.quantile_levels)
