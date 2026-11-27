"""Zitra v3 — 100% Apache-2.0 Probabilistic Time-Series Foundation Models (Zitra-Titan-v3 & Zitra-Core-v3)."""

from .forecaster import ZitraForecast, ZitraForecaster
from .holidays import HOLIDAY_COVARIATE_NAMES, build_holiday_covariates

__version__ = "0.3.2"
__all__ = [
    "ZitraForecaster",
    "ZitraForecast",
    "HOLIDAY_COVARIATE_NAMES",
    "build_holiday_covariates",
    "__version__",
]
