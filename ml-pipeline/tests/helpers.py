"""Test helpers.

IMPORTANT: every number in the tests only exercises code paths (for example a
straight temperature ramp whose slope the code must recover exactly). None of it
is data, a project setting, an estimate or a result, and nothing here is written
to the project's data, models or results folders.
"""
from __future__ import annotations

import copy
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from thermalml.config import Config
from thermalml.io import TABLE_4_5_COLUMNS

ROOT = Path(__file__).resolve().parents[1]


def frame(n: int, run_id: str = "T1", temp=None, **cols) -> pd.DataFrame:
    """A Table 4.5-shaped frame with n rows at 1 s; columns default to constants."""
    t0 = dt.datetime(2000, 1, 1, tzinfo=dt.timezone.utc)
    el = np.arange(n, dtype=float)
    d = {c: np.nan for c in TABLE_4_5_COLUMNS}
    d.update(run_id=run_id, timestamp=[(t0 + dt.timedelta(seconds=float(e))).isoformat() for e in el],
             elapsed_s=el, control_mode="M2",
             soc_temp_c=el * 0.1 if temp is None else temp,
             pcm_t1_c=1.0, pcm_t2_c=1.0, pcm_t3_c=1.0, ambient_temp_c=1.0, cpu_usage_pct=1.0, cpu_freq_mhz=1.0,
             throttle_flags=0, input_power_w=1.0, fan_command_pct=1.0, cpu_freq_limit_mhz=1.0,
             workload_profile="high_performance", inference_fps=1.0, inference_latency_ms=1.0,
             sensor_valid="true", safety_override="false")
    d.update(cols)
    return pd.DataFrame(d)[TABLE_4_5_COLUMNS]


def write_run(df: pd.DataFrame, folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / f"{df['run_id'].iloc[0]}.csv"
    df.to_csv(p, index=False)
    return p


def make_test_config(tmp: Path | None = None, **overrides) -> Config:
    """The real config file with its nulls replaced by test-only values."""
    data = yaml.safe_load(open(ROOT / "config/pipeline_config.yaml"))
    data = copy.deepcopy(data)
    fill = {
        "missing_data.max_missing_fraction": 0.5,
        "predictor.event_kinds_that_break_target": ["schedule_change"],
        "predictor.slope_windows_s": [3, 6],
        "predictor.primary_horizon_s": 10,
        "baselines.linear_trend_window_s": 5,
        "xgboost.grid": {"max_depth": [2], "n_estimators": [20]},
        "xgboost.early_stopping_rounds": 5,
        "rnn.grid": {"hidden_units": [16], "layers": [1], "dropout": [0.0]},
        "rnn.learning_rate": 0.01, "rnn.batch_size": 64, "rnn.max_epochs": 2,
        "rnn.early_stopping_patience": 1, "rnn.seed": 0,
        "evaluation.bootstrap_resamples": 200, "evaluation.confidence_level": 0.9, "evaluation.seed": 0,
    }
    fill.update(overrides)
    for k, v in fill.items():
        node = data
        parts = k.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = v
    return Config(data, None, tmp or ROOT)
