"""Zitra v3 — 100% Apache-2.0 Test-Time Adaptive Probabilistic Time-Series Foundation Model."""

from .forecaster import ZitraForecast, ZitraForecaster
from .holidays import HOLIDAY_COVARIATE_NAMES, build_holiday_covariates

__version__ = "0.3.0"
__all__ = [
    "ZitraForecaster",
    "ZitraForecast",
    "HOLIDAY_COVARIATE_NAMES",
    "build_holiday_covariates",
    "__version__",
]
