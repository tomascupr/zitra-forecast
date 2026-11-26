"""Minimal official GIFT-Eval replication script for Zitra v2 checkpoints.

Usage:
    pip install "zitra-forecast[all] @ git+https://github.com/tomascupr/zitra-forecast.git" gift-eval
    GIFT_EVAL=/path/to/gift_eval python gift_eval.py --model rohlik/Zitra-Titan-585M --out all_results.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from gluonts.ev.metrics import MAE, MAPE, MASE, MSE, MSIS, ND, NRMSE, RMSE, SMAPE, MeanWeightedSumQuantileLoss
from gluonts.model.evaluation import evaluate_model
from gluonts.model.forecast import QuantileForecast
from gift_eval.data import Dataset
from zitra import ZitraForecaster

DECILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
METRICS = [
    MSE(forecast_type="mean"),
    MSE(forecast_type=0.5),
    MAE(),
    MASE(),
    MAPE(),
    SMAPE(),
    MSIS(),
    RMSE(),
    NRMSE(),
    ND(),
    MeanWeightedSumQuantileLoss(quantile_levels=DECILES),
]


class ZitraGluonTSPredictor:
    def __init__(self, forecaster: ZitraForecaster, prediction_length: int, batch_size: int = 48):
        self.forecaster = forecaster
        self.prediction_length = int(prediction_length)
        self.batch_size = int(batch_size)

    def predict(self, dataset):
        entries = list(dataset)
        H = self.prediction_length
        for i0 in range(0, len(entries), self.batch_size):
            batch = entries[i0 : i0 + self.batch_size]
            targets = [np.asarray(e["target"], dtype=np.float32) for e in batch]
            is_mv = targets[0].ndim == 2
            max_t = max(t.shape[-1] for t in targets)
            V = targets[0].shape[0] if is_mv else 1
            ctx = np.full((len(batch), V, max_t), np.nan, dtype=np.float32)
            for bi, t in enumerate(targets):
                arr = t if is_mv else t[None, :]
                ctx[bi, :, max_t - arr.shape[-1] :] = arr
            fc = self.forecaster.forecast(ctx, prediction_length=H)
            q9 = np.stack([fc.quantile(q) for q in DECILES], axis=0)  # (9, B, V, H)
            for bi, entry in enumerate(batch):
                arr = q9[:, bi] if is_mv else q9[:, bi, 0, :]
                if is_mv:
                    arr = np.transpose(arr, (0, 2, 1))  # (9, H, V)
                yield QuantileForecast(
                    forecast_arrays=arr,
                    forecast_keys=[str(q) for q in DECILES],
                    start_date=entry["start"] + targets[bi].shape[-1],
                    item_id=entry.get("item_id"),
                )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tomcupr/Zitra-Titan-585M")
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--dataset", default="m4_hourly")
    ap.add_argument("--term", default="short")
    args = ap.parse_args()

    forecaster = ZitraForecaster.from_pretrained(args.model)
    ds = Dataset(name=args.dataset, term=args.term, to_univariate=False)
    pred = ZitraGluonTSPredictor(forecaster, ds.prediction_length, batch_size=args.batch_size)
    res = evaluate_model(
        pred,
        test_data=ds.test_data,
        metrics=METRICS,
        axis=None,
        mask_invalid_label=True,
        allow_nan_forecast=False,
        seasonality=ds.freq,
    )
    print(f"{args.dataset}/{args.term}: MASE={res['MASE[0.5]'].iloc[0]:.4f} CRPS={res['mean_weighted_sum_quantile_loss'].iloc[0]:.4f}")


if __name__ == "__main__":
    main()
