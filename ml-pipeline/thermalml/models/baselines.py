"""Baselines for the thermal predictor.

Persistence (Azim's brief): the future temperature equals the current one, so the
predicted change is 0.
Linear trend: the least-squares slope of the last N seconds of soc_temp_c,
extrapolated over H. N comes from the config (baselines.linear_trend_window_s).
"""
from __future__ import annotations

import numpy as np

from ..features import Samples, ls_slope


def persistence(s: Samples) -> np.ndarray:
    return np.zeros(len(s.y_delta))


def linear_trend(s: Samples, period: float) -> np.ndarray:
    if s.recent_temp is None:
        raise ValueError("build_samples was called without trend_window_s")
    return ls_slope(s.recent_temp, period) * s.horizon_s
