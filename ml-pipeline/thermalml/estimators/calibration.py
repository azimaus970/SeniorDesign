"""Calibration package and reference capacity state z (Part III 4.2.5 stage D, 4.3.3.1 A).

File formats (column names proposed here; they must be agreed with whoever runs
the stage D calibration before it starts):

calibration/pcm_capacity_reference.csv, one row per cartridge and fan level:
    cartridge_id, fan_command_pct, euse_mean_j, euse_std_j, uncertainty_j,
    c_fix_j_per_k, r_loss_k_per_w, r_proc_k_per_w
  r_loss_k_per_w is the operational cartridge-to-air resistance on T3-T4 and
  r_proc_k_per_w the processor-side resistance from the Pi steady-state run (E0).

calibration/rig_loss.csv, the measured calibration-rig loss points:
    fan_command_pct, delta_t_fix_amb_k, p_loss_w
  Ploss at a sample is interpolated between measured points of the same fan
  level; values outside the measured range are refused, not extrapolated.

calibration/calibration_NNN.csv, one heater cycle per file:
    cycle_id, elapsed_s, phase (melt | solidify), heater_voltage_v,
    heater_current_a, fixture_temp_c, pcm_t1_c, pcm_t2_c, pcm_t3_c,
    ambient_temp_c, fan_command_pct
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REFERENCE_COLUMNS = ["cartridge_id", "fan_command_pct", "euse_mean_j", "euse_std_j", "uncertainty_j",
                     "c_fix_j_per_k", "r_loss_k_per_w", "r_proc_k_per_w"]
RIG_LOSS_COLUMNS = ["fan_command_pct", "delta_t_fix_amb_k", "p_loss_w"]
CYCLE_COLUMNS = ["cycle_id", "elapsed_s", "phase", "heater_voltage_v", "heater_current_a", "fixture_temp_c",
                 "pcm_t1_c", "pcm_t2_c", "pcm_t3_c", "ambient_temp_c", "fan_command_pct"]


def _check(df: pd.DataFrame, cols: list[str], path) -> pd.DataFrame:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    return df


def load_reference(path: Path, cartridge_id: str) -> pd.DataFrame:
    df = _check(pd.read_csv(path, dtype={"cartridge_id": str}), REFERENCE_COLUMNS, path)
    df = df[df["cartridge_id"] == str(cartridge_id)]
    if df.empty:
        raise ValueError(f"{path}: no calibration rows for cartridge {cartridge_id}")
    return df.set_index("fan_command_pct")


def load_rig_loss(path: Path) -> pd.DataFrame:
    return _check(pd.read_csv(path), RIG_LOSS_COLUMNS, path)


def load_cycle(path: Path) -> pd.DataFrame:
    df = _check(pd.read_csv(path, dtype={"cycle_id": str, "phase": str}), CYCLE_COLUMNS, path)
    if df["cycle_id"].nunique() != 1 or df["phase"].nunique() != 1:
        raise ValueError(f"{path}: one cycle and one phase per file")
    if not set(df["phase"]) <= {"melt", "solidify"}:
        raise ValueError(f"{path}: phase must be 'melt' or 'solidify'")
    return df.sort_values("elapsed_s").reset_index(drop=True)


def rig_loss_w(rig: pd.DataFrame, fan: np.ndarray, dt_k: np.ndarray) -> np.ndarray:
    out = np.full(len(fan), np.nan)
    for level in np.unique(fan[np.isfinite(fan)]):
        pts = rig[rig["fan_command_pct"] == level].sort_values("delta_t_fix_amb_k")
        sel = fan == level
        if len(pts) < 2:
            raise ValueError(f"rig loss has fewer than two measured points at fan level {level}")
        x = dt_k[sel]
        lo, hi = pts["delta_t_fix_amb_k"].min(), pts["delta_t_fix_amb_k"].max()
        if np.any((x < lo) | (x > hi)):
            raise ValueError(f"fixture-ambient difference outside the measured rig-loss range "
                             f"[{lo}, {hi}] K at fan level {level}")
        out[sel] = np.interp(x, pts["delta_t_fix_amb_k"], pts["p_loss_w"])
    return out


def actual_dt(elapsed: np.ndarray) -> np.ndarray:
    """Actual interval to the previous sample (Part III: actual sample intervals are used)."""
    d = np.diff(elapsed, prepend=elapsed[0])
    if np.any(d < 0):
        raise ValueError("elapsed_s is not increasing")
    return d


def cycle_energy(cycle: pd.DataFrame, rig: pd.DataFrame, c_fix: float) -> np.ndarray:
    """Cumulative heat into the cartridge (melt) or out of it (solidify), in J, at each sample.

    melt:     sum_k (P_heater - P_loss) dt_k - C_fix * (T_fix,k - T_fix,0)
    solidify: sum_k P_loss dt_k + C_fix * (T_fix,k - T_fix,0)     (heater off)
    (Part III 4.2.5 and 4.3.3.1 A, after Zsembinszki et al., 2020.)
    """
    dt = actual_dt(cycle["elapsed_s"].to_numpy(float))
    tfix = cycle["fixture_temp_c"].to_numpy(float)
    ploss = rig_loss_w(rig, cycle["fan_command_pct"].to_numpy(float),
                       tfix - cycle["ambient_temp_c"].to_numpy(float))
    dfix = c_fix * (tfix - tfix[0])
    if cycle["phase"].iloc[0] == "melt":
        p_heat = cycle["heater_voltage_v"].to_numpy(float) * cycle["heater_current_a"].to_numpy(float)
        return np.cumsum((p_heat - ploss) * dt) - dfix
    return np.cumsum(ploss * dt) + dfix


def euse_from_melt_cycle(cycle: pd.DataFrame, rig: pd.DataFrame, c_fix: float) -> float:
    """Usable capacity from one melt cycle run from the regenerated to the depleted endpoint."""
    if cycle["phase"].iloc[0] != "melt":
        raise ValueError("Euse is computed from melt cycles")
    return float(cycle_energy(cycle, rig, c_fix)[-1])


def reference_z(cycle: pd.DataFrame, rig: pd.DataFrame, c_fix: float, euse: float) -> np.ndarray:
    """Reference capacity state z for every sample of one heater cycle.

    Melt cycles start at the regenerated endpoint (z = 1); solidification cycles
    start at the depleted endpoint (z = 0). Values are not clipped, so any
    disagreement with Euse stays visible.
    """
    e = cycle_energy(cycle, rig, c_fix)
    if cycle["phase"].iloc[0] == "melt":
        return 1.0 - e / euse
    return e / euse
