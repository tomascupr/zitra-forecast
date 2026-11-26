"""Unified dead-simple user interface for Zitra v3 (100% Pure Apache-2.0):
  - Zitra-Titan-v3 (Flagship 1.06B 4-architecture 16K test-time adaptive forecaster)
  - Zitra-Core-v3  (Compact 256M fast distilled forecaster — 17.5 ms GPU / 87 ms CPU)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

EXPERT_NAMES = ("t0", "chronos2", "patchtst", "iamtime")


@dataclass
class ZitraForecast:
    """Container returned by ``ZitraForecaster.forecast()``."""

    quantiles: torch.Tensor  # (B, V, H, Q)
    quantile_levels: tuple[float, ...]
    routing_weights: np.ndarray | None = None  # (B, V, 4) on Zitra-Titan-v3
    expert_medians: dict[str, np.ndarray] | None = None  # optional per-expert medians (B, V, H)

    @property
    def median(self) -> np.ndarray:
        return self.quantile(0.5)

    def quantile(self, level: float) -> np.ndarray:
        idx = min(range(len(self.quantile_levels)), key=lambda i: abs(self.quantile_levels[i] - level))
        if abs(self.quantile_levels[idx] - level) > 1e-4:
            raise KeyError(f"Quantile level {level} not in {self.quantile_levels}")
        return self.quantiles[..., idx].detach().cpu().float().numpy()

    def routing_summary(self) -> dict[str, float] | None:
        if self.routing_weights is None:
            return None
        names = tuple(self.expert_medians.keys()) if self.expert_medians else EXPERT_NAMES
        avg = self.routing_weights.reshape(-1, len(names)).mean(axis=0)
        return {name: round(float(val), 4) for name, val in zip(names, avg)}


def _resolve_checkpoint_dir(repo_id_or_path: str | Path, revision: str | None = None, token: str | None = None) -> Path:
    p = Path(repo_id_or_path)
    if p.exists():
        return p
    from huggingface_hub import snapshot_download

    local_dir = snapshot_download(
        repo_id=str(repo_id_or_path),
        revision=revision,
        token=token,
        allow_patterns=["*.json", "*.safetensors", "*.py", "_iamtime/*.py"],
    )
    return Path(local_dir)


def _detect_variant(ckpt_dir: Path) -> str:
    if (ckpt_dir / "patchtst_config.json").exists() or (ckpt_dir / "zitra_v3_forecaster.py").exists():
        return "titan_v3"
    if (ckpt_dir / "zitra_core_v3_forecaster.py").exists():
        return "core_v3"
    frozen_p = ckpt_dir / "frozen.json"
    if frozen_p.exists():
        meta = json.loads(frozen_p.read_text())
        name = str(meta.get("model", meta.get("name", ""))).lower()
        if "core" in name:
            return "core_v3"
        if "v3" in name or "titan" in name:
            return "titan_v3"
    if (ckpt_dir / "t0_config.json").exists() and not (ckpt_dir / "chronos_config.json").exists():
        return "core_v3"
    raise ValueError(f"Unrecognized Zitra v3 checkpoint directory: {ckpt_dir}")


class ZitraForecaster:
    """Unified zero-friction loader and predictor for Zitra v3 (100% Apache-2.0).

    Examples
    --------
    >>> from zitra import ZitraForecaster
    >>> model = ZitraForecaster.from_pretrained("tomcupr/Zitra-Titan-v3")
    >>> fc = model.forecast(np.cumsum(np.random.randn(256)), prediction_length=48)
    >>> fc.median.shape
    (1, 1, 48)
    """

    def __init__(self, backend: Any, variant: str, model_id: str):
        self.backend = backend
        self.variant = variant
        self.model_id = model_id
        self.quantile_levels = tuple(float(q) for q in backend.quantile_levels)
        self.max_context = int(backend.max_context)

    @classmethod
    def from_pretrained(
        cls,
        repo_id_or_path: str | Path,
        device: str | None = None,
        revision: str | None = None,
        token: str | None = None,
    ) -> "ZitraForecaster":
        """Load a Zitra checkpoint from Hugging Face Hub or a local directory."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt_dir = _resolve_checkpoint_dir(repo_id_or_path, revision=revision, token=token)
        variant = _detect_variant(ckpt_dir)

        if variant == "titan_v3":
            from .backends.titan_v3 import ZitraV3Forecaster

            backend = ZitraV3Forecaster.from_pretrained(ckpt_dir, device=device)
        elif variant == "core_v3":
            from .backends.core_v3 import ZitraCoreV3Forecaster

            backend = ZitraCoreV3Forecaster.from_pretrained(ckpt_dir, device=device)
        else:
            raise ValueError(f"Unsupported Zitra variant: {variant}")

        return cls(backend=backend, variant=variant, model_id=str(repo_id_or_path))

    @torch.inference_mode()
    def forecast(
        self,
        context: np.ndarray | torch.Tensor | Sequence[float],
        prediction_length: int,
        *,
        observation_mask: np.ndarray | torch.Tensor | None = None,
        past_covariates: Any = None,
        known_covariates: Any = None,
        categorical_past_covariates: Any = None,
        categorical_known_covariates: Any = None,
        cross_learning: str | bool | None = None,
        fast: bool = False,
    ) -> ZitraForecast:
        """Forecast 1D `(T,)`, 2D `(B, T)`, or 3D `(B, V, T)` numeric context arrays."""
        arr = np.asarray(context, dtype=np.float32) if not isinstance(context, torch.Tensor) else context
        if arr.ndim == 1:
            arr = arr[None, None, :]
        elif arr.ndim == 2:
            arr = arr[:, None, :]
        elif arr.ndim != 3:
            raise ValueError(f"Expected 1D, 2D, or 3D context array, got shape {tuple(arr.shape)}")

        kwargs: dict[str, Any] = {}
        if observation_mask is not None:
            kwargs["observation_mask"] = observation_mask
        if past_covariates is not None:
            kwargs["past_covariates"] = past_covariates
        if known_covariates is not None:
            kwargs["known_covariates"] = known_covariates
        if categorical_past_covariates is not None:
            kwargs["categorical_past_covariates"] = categorical_past_covariates
        if categorical_known_covariates is not None:
            kwargs["categorical_known_covariates"] = categorical_known_covariates
        if cross_learning is not None and self.variant in ("titan_v3", "core_v3"):
            kwargs["cross_learning"] = cross_learning
        if fast and self.variant in ("titan_v3", "core_v3"):
            kwargs["fast"] = True

        out = self.backend.forecast(arr, prediction_length=int(prediction_length), **kwargs)
        routing_w = getattr(self.backend, "last_routing_weights", None)
        if routing_w is not None and routing_w.ndim == 2:
            routing_w = routing_w.reshape(arr.shape[0], arr.shape[1], routing_w.shape[-1])
        expert_med = getattr(self.backend, "last_expert_medians", None)
        return ZitraForecast(
            quantiles=out.quantiles,
            quantile_levels=tuple(float(q) for q in out.quantile_levels),
            routing_weights=routing_w,
            expert_medians=expert_med,
        )

    def predict_df(
        self,
        df: pd.DataFrame,
        prediction_length: int,
        *,
        id_column: str = "item_id",
        timestamp_column: str = "timestamp",
        target: str = "target",
        quantile_levels: Sequence[float] = (0.1, 0.5, 0.9),
        future_df: pd.DataFrame | None = None,
        past_covariate_columns: Sequence[str] | None = None,
        known_covariate_columns: Sequence[str] | None = None,
        country: str | None = None,
        country_column: str | None = None,
        fast: bool = False,
        batch_size: int = 64,
    ) -> pd.DataFrame:
        """Chronos-2-style DataFrame API returning tidy predictions with timestamps and quantiles.

        Pass ``country="CZ"`` (or regional ISO codes like ``"DE-BY"``, ``"DE-BE"``, ``"US-CA"``)
        or ``country_column="region"`` to automatically synthesize 5 known-future public holiday
        and calendar covariates (``is_public_holiday``, ``is_pre_holiday``, ``is_post_holiday``,
        ``dow_sin``, ``dow_cos``) across both history and forecast horizon.
        """
        work = df.copy()
        if id_column not in work.columns:
            work[id_column] = "series_0"
        work[timestamp_column] = pd.to_datetime(work[timestamp_column])
        work = work.sort_values([id_column, timestamp_column])

        fwork = None
        if future_df is not None:
            fwork = future_df.copy()
            if id_column not in fwork.columns:
                fwork[id_column] = "series_0"
            fwork[timestamp_column] = pd.to_datetime(fwork[timestamp_column])
            fwork = fwork.sort_values([id_column, timestamp_column])

        past_cols = list(past_covariate_columns or [])
        known_cols = list(known_covariate_columns or [])
        if known_cols and fwork is None:
            raise ValueError("`future_df` must be provided when `known_covariate_columns` is specified.")

        use_holidays = (country is not None) or (country_column is not None)
        if use_holidays:
            from .holidays import build_holiday_covariates

        grouped = list(work.groupby(id_column, sort=False))
        f_groups = dict(list(fwork.groupby(id_column, sort=False))) if fwork is not None else {}

        H = int(prediction_length)
        rows_out: list[pd.DataFrame] = []
        n_hol = 5 if use_holidays else 0
        n_known_total = len(known_cols) + n_hol

        for i0 in range(0, len(grouped), batch_size):
            chunk = grouped[i0 : i0 + batch_size]
            B = len(chunk)
            max_t = max(len(g) for _, g in chunk)
            ctx = np.full((B, 1, max_t), np.nan, dtype=np.float32)
            p_cov = np.full((B, len(past_cols), max_t), np.nan, dtype=np.float32) if past_cols else None
            k_cov = np.full((B, n_known_total, max_t + H), np.nan, dtype=np.float32) if n_known_total > 0 else None

            future_ts_list: list[pd.DatetimeIndex] = []
            item_ids: list[Any] = []

            for bi, (item_id, g) in enumerate(chunk):
                item_ids.append(item_id)
                t_len = len(g)
                off = max_t - t_len
                ctx[bi, 0, off:] = g[target].to_numpy(dtype=np.float32)
                if p_cov is not None:
                    for ci, col in enumerate(past_cols):
                        p_cov[bi, ci, off:] = g[col].to_numpy(dtype=np.float32)

                ts = pd.DatetimeIndex(g[timestamp_column])
                fg = f_groups.get(item_id)
                if fg is not None and len(fg) >= H:
                    fts = pd.DatetimeIndex(fg[timestamp_column].iloc[:H])
                else:
                    step = (ts[-1] - ts[-2]) if len(ts) >= 2 else pd.Timedelta(days=1)
                    fts = pd.date_range(start=ts[-1] + step, periods=H, freq=step)
                future_ts_list.append(fts)

                if known_cols:
                    if fg is None or len(fg) < H:
                        raise ValueError(f"future_df missing {H} rows for {id_column}={item_id!r}")
                    for ci, col in enumerate(known_cols):
                        k_cov[bi, ci, off:max_t] = g[col].to_numpy(dtype=np.float32)
                        k_cov[bi, ci, max_t : max_t + H] = fg[col].iloc[:H].to_numpy(dtype=np.float32)

                if use_holidays:
                    c_val = str(g[country_column].iloc[-1]) if country_column is not None else str(country)
                    full_ts = ts.append(fts)
                    hol_mat = build_holiday_covariates(full_ts, c_val)  # (5, t_len + H)
                    k_cov[bi, len(known_cols) :, off : max_t + H] = hol_mat

            fc = self.forecast(
                ctx,
                prediction_length=H,
                past_covariates=p_cov,
                known_covariates=k_cov,
                fast=fast,
            )

            for bi, item_id in enumerate(item_ids):
                rec: dict[str, Any] = {
                    id_column: item_id,
                    timestamp_column: future_ts_list[bi],
                    "predictions": fc.median[bi, 0],
                }
                for q in quantile_levels:
                    rec[f"{q:g}"] = fc.quantile(float(q))[bi, 0]
                if fc.routing_weights is not None:
                    w = fc.routing_weights[bi, 0]
                    names = tuple(fc.expert_medians.keys()) if fc.expert_medians else EXPERT_NAMES
                    for ei, ename in enumerate(names):
                        col_key = "weight_" + ename.replace("-", "_")
                        rec[col_key] = float(w[ei])
                rows_out.append(pd.DataFrame(rec))

        return pd.concat(rows_out, ignore_index=True) if rows_out else pd.DataFrame()
