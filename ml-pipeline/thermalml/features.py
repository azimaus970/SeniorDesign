"""Turn validated run logs into predictor samples (Part III 4.3.3.1 B/C, 4.3.3.2).

For one run, one horizon H and one window W this module produces, for every
eligible time step t:
  * the target, soc_temp_c(t+H) - soc_temp_c(t) (the Part III target rewritten
    as a change; predicted_temp_c = soc_temp_c(t) + predicted change),
  * the flat feature vector used by XGBoost,
  * the W x F window used by GRU/LSTM (built only when asked for).

A row is eligible only if
  1. soc_temp_c is measured (not held) at t and at t+H, and t+H is inside the run;
  2. no command, safety override or listed event changes between t and t+H;
  3. after the Part III hold rule (last valid value for at most
     hold_max_periods), every input in the window is present, and the share of
     held values in the window is at most max_missing_fraction.
Within one (H, W) every model and baseline sees the same eligible rows; when settings are
compared, predictors.common_origins() restricts scoring to the origins eligible for every
compared window and variant, so all compared errors use the same target instances.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from .config import Config
from .io import throttle_bits

Z_COLUMN = "z_hat_offline"   # estimator output applied offline (Part III 4.3.3.1 B)


def steps(seconds: float, period: float, what: str) -> int:
    n = seconds / period
    if abs(n - round(n)) > 1e-9:
        raise ValueError(f"{what} = {seconds} s is not a whole number of {period} s periods")
    return int(round(n))


def regularize(run: pd.DataFrame, period: float) -> pd.DataFrame:
    """Place samples on the control-period grid k = round(elapsed_s / period).

    Missing cycles become empty rows (row_present = False). Two samples landing on
    the same grid point is an error, not something to average.
    """
    k = np.rint(run["elapsed_s"].to_numpy() / period).astype(np.int64)
    if len(np.unique(k)) != len(k):
        raise ValueError(f"run {run['run_id'].iloc[0]}: two samples fall on the same {period} s grid point")
    df = run.copy()
    df["k"] = k
    df["row_present"] = True
    full = pd.DataFrame({"k": np.arange(k.min(), k.max() + 1)})
    df = full.merge(df, on="k", how="left")
    df["row_present"] = df["row_present"].fillna(False).astype(bool)
    df["run_id"] = run["run_id"].iloc[0]
    return df


def mark_invalid_rows(df: pd.DataFrame, fields: list[str]) -> pd.DataFrame:
    """Rows where the logger set sensor_valid = False are treated as missing measurements."""
    df = df.copy()
    bad = df["sensor_valid"].fillna(True).eq(False).to_numpy() if "sensor_valid" in df else np.zeros(len(df), bool)
    for f in fields:
        df.loc[bad, f] = np.nan
    return df


def hold(df: pd.DataFrame, fields: list[str], max_periods: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Part III 4.3.3.1 C: hold the last valid value for at most max_periods, flagged."""
    held = df[fields].ffill(limit=max_periods)
    flags = df[fields].isna() & held.notna()
    return held, flags


def ls_slope(windows: np.ndarray, period: float) -> np.ndarray:
    """Least-squares slope (per second) along the last axis."""
    n = windows.shape[-1]
    t = (np.arange(n) - (n - 1) / 2.0) * period
    return (windows * t).sum(axis=-1) / (t * t).sum()


@dataclass
class Samples:
    run_id: str
    horizon_s: float
    window_s: float
    variant: str
    elapsed_s: np.ndarray            # time t of each sample
    current_temp: np.ndarray         # soc_temp_c(t)
    y_delta: np.ndarray              # soc_temp_c(t+H) - soc_temp_c(t)
    flat: pd.DataFrame               # XGBoost features
    seq: np.ndarray | None = None    # (n, W, F) raw (un-normalized) windows
    seq_fields: list[str] = field(default_factory=list)
    recent_temp: np.ndarray | None = None   # last soc_temp_c values for the trend baseline
    coverage: dict = field(default_factory=dict)


def _profile_onehot(series: pd.Series, profiles: list[str]) -> pd.DataFrame:
    unknown = sorted(set(series.dropna()) - set(profiles))
    if unknown:
        raise ValueError(f"unknown workload_profile values {unknown}; expected {profiles}")
    out = {}
    for p in profiles:
        col = (series == p).astype(float)
        col[series.isna()] = np.nan
        out[f"profile_{p}"] = col
    return pd.DataFrame(out, index=series.index)


def _unchanged_mask(df: pd.DataFrame, h: int, cmd_cols: list[str], drop_on_safety: bool,
                    event_steps: np.ndarray) -> np.ndarray:
    """True at t when nothing that changes future heat flow happens in [t, t+H]."""
    n = len(df)
    ok = np.ones(n, bool)
    ok[n - h:] = False
    for c in cmd_cols:
        v = df[c]
        codes = pd.factorize(v, use_na_sentinel=True)[0].astype(np.int64)  # NaN -> -1
        changed = np.zeros(n, bool)
        changed[1:] = codes[1:] != codes[:-1]
        unknown = codes < 0
        bad = changed | unknown
        # any change at steps t+1..t+h, or unknown command at t..t+h
        cs = np.concatenate([[0], np.cumsum(changed)])
        cu = np.concatenate([[0], np.cumsum(unknown)])
        idx = np.arange(n - h)
        ok[: n - h] &= (cs[idx + h + 1] - cs[idx + 1]) == 0
        ok[: n - h] &= (cu[idx + h + 1] - cu[idx]) == 0
    if drop_on_safety and "safety_override" in df:
        s = df["safety_override"].fillna(False).astype(bool).to_numpy()
        cs = np.concatenate([[0], np.cumsum(s)])
        idx = np.arange(n - h)
        ok[: n - h] &= (cs[idx + h + 1] - cs[idx]) == 0
    for e in event_steps:
        lo = max(0, e - h)          # samples whose interval [t, t+h] contains e
        ok[lo: e + 1] = False
    return ok


def build_samples(run: pd.DataFrame, cfg: Config, horizon_s: float, window_s: float,
                  variant: str = "temperature_only", events: pd.DataFrame | None = None,
                  with_sequences: bool = False, trend_window_s: float | None = None) -> Samples:
    if variant not in ("temperature_only", "pcm_aware"):
        raise ValueError("variant must be 'temperature_only' (M3) or 'pcm_aware' (M4)")
    period = float(cfg.require("timing.sampling_period_s"))
    hold_n = int(cfg.require("missing_data.hold_max_periods"))
    max_missing = float(cfg.require("missing_data.max_missing_fraction"))
    numeric = list(cfg.require("predictor.numeric_fields"))
    bits = list(cfg.require("predictor.throttle_bits"))
    profiles = list(cfg.require("predictor.workload_profiles"))
    target = cfg.require("predictor.target")
    cmd_cols = list(cfg.require("predictor.command_columns"))
    break_kinds = cfg.require("predictor.event_kinds_that_break_target")
    h = steps(horizon_s, period, "horizon")
    w = steps(window_s, period, "window")

    df = regularize(run, period)
    measured = numeric + ["throttle_flags"]
    if variant == "pcm_aware":
        if Z_COLUMN not in df:
            raise ValueError(f"pcm_aware variant needs column '{Z_COLUMN}' from the selected estimator "
                             "(run thermalml.estimators.apply first)")
        measured.append(Z_COLUMN)
    df = mark_invalid_rows(df, measured)

    raw_temp = df[target].to_numpy(float)
    bit_df = throttle_bits(df["throttle_flags"], bits)
    prof_df = _profile_onehot(df["workload_profile"], profiles)
    base = pd.concat([df[numeric], bit_df, prof_df] + ([df[[Z_COLUMN]]] if variant == "pcm_aware" else []), axis=1)
    fields = list(base.columns)
    held, held_flag = hold(base, fields, hold_n)
    H = held.to_numpy(float)                      # (n, F)
    n = len(df)

    # 1. target
    valid_t = np.zeros(n, bool)
    y = np.full(n, np.nan)
    idx = np.arange(n - h)
    y[idx] = raw_temp[idx + h] - raw_temp[idx]
    valid_t[idx] = np.isfinite(y[idx])
    # 2. unchanged commands
    ev_steps = np.array([], dtype=np.int64)
    if events is not None and len(events):
        sel = events[events["event"].isin(break_kinds)]
        ev_steps = np.rint(sel["elapsed_s"].to_numpy() / period).astype(np.int64) - int(df["k"].iloc[0])
        ev_steps = ev_steps[(ev_steps >= 0) & (ev_steps < n)]
    unchanged = _unchanged_mask(df, h, cmd_cols, bool(cfg.get("predictor.drop_on_safety_override", True)), ev_steps)
    # 3. window completeness after hold, and share of held values
    ok_window = np.zeros(n, bool)
    if n >= w:
        complete = np.isfinite(H).all(axis=1)
        cc = np.concatenate([[0], np.cumsum(complete)])
        hf = held_flag.to_numpy().any(axis=1)
        ch = np.concatenate([[0], np.cumsum(hf)])
        end = np.arange(w - 1, n)
        ok_window[end] = ((cc[end + 1] - cc[end + 1 - w]) == w) & \
                         ((ch[end + 1] - ch[end + 1 - w]) / w <= max_missing)
    eligible = valid_t & unchanged & ok_window
    rows = np.flatnonzero(eligible)

    # flat features (XGBoost), Part III 4.3.3.1 B
    lags = [steps(l, period, "lag") for l in cfg.require("predictor.lags_s")]
    lags = [l for l in lags if l <= w - 1]        # the buffer only holds W seconds
    slopes = [steps(s, period, "slope window") for s in cfg.require("predictor.slope_windows_s")]
    slopes = [s for s in slopes if 2 <= s <= w]
    ti = fields.index(target)
    feat = {}
    for j, f in enumerate(fields):
        feat[f] = H[rows, j]
    for l in lags:
        feat[f"{target}_lag{l * period:g}s"] = H[rows - l, ti]
    if len(rows):
        win = sliding_window_view(H, w, axis=0)[rows - w + 1]      # (m, F, w)
        tw = win[:, ti, :]
        for s in slopes:
            feat[f"{target}_slope{s * period:g}s"] = ls_slope(tw[:, -s:], period)
        feat[f"{target}_max_W"] = tw.max(axis=1)
        for j, f in enumerate(numeric):
            feat[f"{f}_mean_W"] = win[:, j, :].mean(axis=1)
        if "input_power_w" in numeric:
            feat["input_energy_J_W"] = win[:, numeric.index("input_power_w"), :].sum(axis=1) * period
    flat = pd.DataFrame(feat)

    seq = None
    if with_sequences and len(rows):
        seq = np.ascontiguousarray(np.transpose(sliding_window_view(H, w, axis=0)[rows - w + 1], (0, 2, 1)))
    recent = None
    if trend_window_s is not None and len(rows):
        tn = steps(trend_window_s, period, "trend window")
        if tn > w:
            raise ValueError("trend baseline window must not exceed the input window")
        recent = sliding_window_view(H[:, ti], tn)[rows - tn + 1]

    coverage = {"rows_on_grid": int(n), "rows_logged": int(df["row_present"].sum()),
                "target_valid": int(valid_t.sum()), "unchanged_commands": int((valid_t & unchanged).sum()),
                "eligible": int(len(rows))}
    return Samples(run_id=str(df["run_id"].iloc[0]), horizon_s=horizon_s, window_s=window_s, variant=variant,
                   elapsed_s=df["elapsed_s"].to_numpy(float)[rows] if len(rows) else np.array([]),
                   current_temp=raw_temp[rows], y_delta=y[rows], flat=flat, seq=seq, seq_fields=fields,
                   recent_temp=recent, coverage=coverage)
