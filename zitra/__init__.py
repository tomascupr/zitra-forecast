"""Zitra v2 — State-of-the-art probabilistic time-series foundation models."""

from .forecaster import ZitraForecast, ZitraForecaster

__version__ = "0.2.0"
__all__ = ["ZitraForecaster", "ZitraForecast", "__version__"]
