"""Estimator features (Part III 4.3.3.1 A).

T1-T3, the T1-T3 gradient, least-squares slopes of each, time each probe has
spent in the transition range since the start of the run or cycle, ambient
temperature, T3-T4 and the applied fan command. Processor temperature, input
power and the E0 output are deliberately excluded (Part III).
The same function serves calibration cycles and operational runs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from ..config import Config
from ..features import ls_slope, steps

T_FIELDS = ["pcm_t1_c", "pcm_t2_c", "pcm_t3_c"]


def estimator_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """df must be on the control-period grid with no gaps in the required fields.

    Rows without full slope history get NaN slopes; the caller decides whether to drop them.
    """
    period = float(cfg.require("timing.sampling_period_s"))
    lo, hi = cfg.require("estimator.transition_range_c")
    slope_s = [steps(s, period, "estimator slope window") for s in cfg.require("estimator.slope_windows_s")]
    amb = cfg.require("estimator.ambient_field")
    fan = cfg.require("estimator.fan_field")
    out = {}
    for f in T_FIELDS:
        out[f] = df[f].to_numpy(float)
    out["grad_t1_t3"] = out["pcm_t1_c"] - out["pcm_t3_c"]
    for f in T_FIELDS:
        x = out[f]
        in_range = ((x >= lo) & (x <= hi)).astype(float)
        out[f"time_in_transition_{f}_s"] = np.cumsum(in_range) * period
        for s in slope_s:
            col = np.full(len(x), np.nan)
            if len(x) >= s:
                col[s - 1:] = ls_slope(sliding_window_view(x, s), period)
            out[f"{f}_slope{s * period:g}s"] = col
    out[amb] = df[amb].to_numpy(float)
    out["t3_minus_t4"] = out["pcm_t3_c"] - out[amb]
    out[fan] = df[fan].to_numpy(float)
    return pd.DataFrame(out, index=df.index)
