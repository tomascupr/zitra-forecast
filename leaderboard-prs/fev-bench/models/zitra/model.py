"""Official autogluon/fev model wrapper for Zitra v2 checkpoints."""

from __future__ import annotations

import time
from typing import Any

import datasets
import fev
import numpy as np
import torch
from zitra import ZitraForecaster

DEFAULT_QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


class ZitraFevModel:
    """Adapter compatible with autogluon/fev benchmark runners."""

    def __init__(self, model_path: str = "tomcupr/Zitra-Titan-585M", batch_size: int = 64, device: str | None = None):
        self.model_path = model_path
        self.batch_size = int(batch_size)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._forecaster: ZitraForecaster | None = None

    @property
    def forecaster(self) -> ZitraForecaster:
        if self._forecaster is None:
            self._forecaster = ZitraForecaster.from_pretrained(self.model_path, device=self.device)
        return self._forecaster

    def predict(self, task: fev.Task) -> tuple[list[datasets.Dataset | datasets.DatasetDict], float]:
        t0 = time.monotonic()
        preds_per_window: list[Any] = []
        quantiles = list(task.quantile_levels or DEFAULT_QUANTILES)
        H = int(task.horizon)

        for window in task.iter_windows():
            past_data, future_data = window.get_input_data()
            targets = [task.target] if isinstance(task.target, str) else list(task.target)
            n_series = len(past_data)
            V = len(targets)

            # Build per-series arrays with optional past & known covariates
            past_cols = list(task.past_dynamic_columns or [])
            known_cols = list(task.known_dynamic_columns or [])

            per_target_out: dict[str, dict[str, list[list[float]]]] = {
                t: {"predictions": [], **{str(q): [] for q in quantiles}} for t in targets
            }

            for i0 in range(0, n_series, self.batch_size):
                i1 = min(n_series, i0 + self.batch_size)
                B = i1 - i0
                rows = [past_data[i] for i in range(i0, i1)]
                frows = [future_data[i] for i in range(i0, i1)]
                max_t = max(len(r[targets[0]]) for r in rows)

                ctx = np.full((B, V, max_t), np.nan, dtype=np.float32)
                p_cov = np.full((B, len(past_cols), max_t), np.nan, dtype=np.float32) if past_cols else None
                k_cov = np.full((B, len(known_cols), max_t + H), np.nan, dtype=np.float32) if known_cols else None

                for bi, (r, fr) in enumerate(zip(rows, frows)):
                    t_len = len(r[targets[0]])
                    off = max_t - t_len
                    for vi, col in enumerate(targets):
                        ctx[bi, vi, off:] = np.asarray(r[col], dtype=np.float32)
                    if p_cov is not None:
                        for ci, col in enumerate(past_cols):
                            vals = np.asarray(r[col], dtype=np.float32)
                            p_cov[bi, ci, off : off + len(vals)] = vals
                    if k_cov is not None:
                        for ci, col in enumerate(known_cols):
                            vp = np.asarray(r[col], dtype=np.float32)
                            vf = np.asarray(fr[col], dtype=np.float32)[:H]
                            k_cov[bi, ci, off : off + len(vp)] = vp
                            k_cov[bi, ci, max_t : max_t + len(vf)] = vf

                fc = self.forecaster.forecast(
                    ctx,
                    prediction_length=H,
                    past_covariates=p_cov,
                    known_covariates=k_cov,
                )
                med = fc.median  # (B, V, H)
                q_map = {q: fc.quantile(float(q)) for q in quantiles}
                for bi in range(B):
                    for vi, tcol in enumerate(targets):
                        per_target_out[tcol]["predictions"].append(med[bi, vi].tolist())
                        for q in quantiles:
                            per_target_out[tcol][str(q)].append(q_map[q][bi, vi].tolist())

            if isinstance(task.target, str):
                preds_per_window.append(datasets.Dataset.from_dict(per_target_out[targets[0]]))
            else:
                preds_per_window.append(
                    datasets.DatasetDict({tcol: datasets.Dataset.from_dict(per_target_out[tcol]) for tcol in targets})
                )

        return preds_per_window, time.monotonic() - t0
