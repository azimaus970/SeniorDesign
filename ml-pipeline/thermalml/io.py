"""Load run logs, run configurations and event logs.

The main per-run CSV follows Part III Table 4.5 exactly. Missing values stay
missing (NaN); nothing is filled here (Part III 4.3.3.1 C).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# Part III Table 4.5, in order.
TABLE_4_5_COLUMNS = [
    "run_id", "timestamp", "elapsed_s", "control_mode", "soc_temp_c", "pcm_t1_c", "pcm_t2_c",
    "pcm_t3_c", "ambient_temp_c", "cpu_usage_pct", "cpu_freq_mhz", "throttle_flags", "input_power_w",
    "pcm_capacity_est", "predicted_temp_c", "fan_command_pct", "cpu_freq_limit_mhz", "workload_profile",
    "inference_fps", "inference_latency_ms", "sensor_valid", "safety_override",
]
STRING_COLUMNS = {"run_id", "timestamp", "control_mode", "workload_profile"}
BOOL_COLUMNS = {"sensor_valid", "safety_override"}
NUMERIC_COLUMNS = [c for c in TABLE_4_5_COLUMNS if c not in STRING_COLUMNS | BOOL_COLUMNS]

# Keys of the per-run configuration file (Part III 4.3.2 and 4.3.3.1 B).
RUN_CONFIG_KEYS = ["run_id", "control_mode", "chamber_setpoint_c", "sampling_period_s",
                   "initial_pcm_state", "cartridge_id", "workload_config", "schedule_id"]


class SchemaError(ValueError):
    pass


def _to_bool(s: pd.Series) -> pd.Series:
    m = {"true": True, "1": True, "false": False, "0": False}
    out = s.astype("string").str.strip().str.lower().map(m)
    return out.astype("boolean")


def load_run(path: str | Path) -> pd.DataFrame:
    """Read one Table 4.5 CSV. Empty cells become NaN / <NA>; nothing is imputed."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    missing = [c for c in TABLE_4_5_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"{path}: missing Table 4.5 columns {missing}")
    for c in NUMERIC_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="raise")
    for c in BOOL_COLUMNS:
        df[c] = _to_bool(df[c])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601")
    if df["run_id"].nunique() != 1:
        raise SchemaError(f"{path}: expected one run_id, found {df['run_id'].unique().tolist()}")
    df = df.sort_values("elapsed_s", kind="stable").reset_index(drop=True)
    if df["elapsed_s"].duplicated().any():
        raise SchemaError(f"{path}: duplicated elapsed_s values")
    return df


def load_run_config(path: str | Path) -> dict:
    with open(path) as f:
        cfg = json.load(f)
    missing = [k for k in RUN_CONFIG_KEYS if k not in cfg]
    if missing:
        raise SchemaError(f"{path}: run configuration lacks {missing}")
    return cfg


def load_events(path: str | Path | None, run: pd.DataFrame) -> pd.DataFrame:
    """Event log with an 'event' column and either 'elapsed_s' or 't_unix'.

    Returns columns elapsed_s, event. A missing file gives an empty frame.
    """
    if path is None or not Path(path).exists():
        return pd.DataFrame({"elapsed_s": pd.Series(dtype=float), "event": pd.Series(dtype=str)})
    ev = pd.read_csv(path, dtype={"event": str})
    if "event" not in ev.columns:
        raise SchemaError(f"{path}: needs an 'event' column")
    if "elapsed_s" not in ev.columns:
        if "t_unix" not in ev.columns:
            raise SchemaError(f"{path}: needs 'elapsed_s' or 't_unix'")
        t0 = run["timestamp"].iloc[0].timestamp() - run["elapsed_s"].iloc[0]
        ev["elapsed_s"] = ev["t_unix"].astype(float) - t0
    return ev[["elapsed_s", "event"]].sort_values("elapsed_s").reset_index(drop=True)


def throttle_bits(flags: pd.Series, bits: list[int]) -> pd.DataFrame:
    """Split vcgencmd get_throttled flags into one 0/1 column per bit; NaN stays NaN."""
    out = {}
    valid = flags.notna()
    ints = flags.fillna(0).to_numpy().astype(np.int64)
    for b in bits:
        col = ((ints >> b) & 1).astype(float)
        col[~valid.to_numpy()] = np.nan
        out[f"throttle_bit{b}"] = col
    return pd.DataFrame(out, index=flags.index)


def load_all_runs(runs_dir: Path, run_ids: list[str] | None = None) -> dict[str, pd.DataFrame]:
    runs = {}
    for p in sorted(Path(runs_dir).glob("*.csv")):
        df = load_run(p)
        rid = df["run_id"].iloc[0]
        if run_ids is None or rid in run_ids:
            runs[rid] = df
    if run_ids is not None:
        absent = sorted(set(run_ids) - set(runs))
        if absent:
            raise FileNotFoundError(f"runs listed in the manifest but not found in {runs_dir}: {absent}")
    return runs
