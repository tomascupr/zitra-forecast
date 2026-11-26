"""Built-in opt-in public holiday and calendar known-future covariate generator.

Generates 5 lightweight, zero-leakage known-future covariates aligned to any
DatetimeIndex (daily or sub-daily) for 240+ countries and regional subdivisions
(e.g. ``"CZ"``, ``"DE-BY"`` for Bavaria/Munich, ``"DE-BE"`` for Berlin,
``"DE-HE"`` for Hesse/Frankfurt, ``"AT"``, ``"HU"``, ``"RO"``, ``"US-CA"``).
"""

from __future__ import annotations

import datetime as dt
import math
from functools import lru_cache
from typing import Iterable

import numpy as np
import pandas as pd

HOLIDAY_COVARIATE_NAMES = (
    "is_public_holiday",
    "is_pre_holiday",
    "is_post_holiday",
    "dow_sin",
    "dow_cos",
)


def parse_country_code(code: str) -> tuple[str, str | None]:
    """Parse ``"DE-BY"`` -> ``("DE", "BY")`` or ``"CZ"`` -> ``("CZ", None)``."""
    raw = str(code).strip().upper().replace("_", "-")
    if "-" in raw:
        country, subdiv = raw.split("-", 1)
        return country.strip(), (subdiv.strip() or None)
    return raw, None


@lru_cache(maxsize=256)
def _holiday_dates_for(country: str, subdiv: str | None, years: tuple[int, ...]) -> frozenset[dt.date]:
    try:
        import holidays as pyholidays
    except ImportError as exc:
        raise ImportError(
            "Built-in public holiday covariates require the `holidays` package. "
            'Install it with: pip install holidays  (or pip install "zitra-forecast[holidays]")'
        ) from exc

    cal = pyholidays.country_holidays(country, subdiv=subdiv, years=list(years))
    return frozenset(cal.keys())


def build_holiday_covariates(
    timestamps: pd.DatetimeIndex | Iterable[pd.Timestamp],
    country: str,
) -> np.ndarray:
    """Return float32 array of shape ``(5, L)`` with known-future holiday & calendar features.

    Channels:
      0. ``is_public_holiday`` — 1.0 on official public holidays for ``country``
      1. ``is_pre_holiday``    — 1.0 on the calendar day immediately preceding a holiday
      2. ``is_post_holiday``   — 1.0 on the calendar day immediately following a holiday
      3. ``dow_sin``           — ``sin(2π * day_of_week / 7)``
      4. ``dow_cos``           — ``cos(2π * day_of_week / 7)``
    """
    ts = pd.DatetimeIndex(timestamps)
    L = len(ts)
    out = np.zeros((5, L), dtype=np.float32)
    if L == 0:
        return out

    c_code, subdiv = parse_country_code(country)
    # Include year-1 and year+1 so Jan 1 pre-holiday and Dec 31 post-holiday never miss boundary dates
    min_y = int(ts.min().year) - 1
    max_y = int(ts.max().year) + 1
    hol_set = _holiday_dates_for(c_code, subdiv, tuple(range(min_y, max_y + 1)))

    one_day = dt.timedelta(days=1)
    dates = ts.date
    out[0] = np.fromiter((1.0 if d in hol_set else 0.0 for d in dates), dtype=np.float32, count=L)
    out[1] = np.fromiter((1.0 if (d + one_day) in hol_set else 0.0 for d in dates), dtype=np.float32, count=L)
    out[2] = np.fromiter((1.0 if (d - one_day) in hol_set else 0.0 for d in dates), dtype=np.float32, count=L)

    dow = ts.dayofweek.to_numpy(dtype=np.float32)
    ang = (2.0 * math.pi / 7.0) * dow
    out[3] = np.sin(ang)
    out[4] = np.cos(ang)
    return out
