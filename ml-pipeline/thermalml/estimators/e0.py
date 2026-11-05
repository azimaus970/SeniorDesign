"""E0: physically informed energy-balance estimator (Part III 4.3.3.1 A).

Each step:
    q_in  = (soc_temp_c - T1) / R_proc
    q_out = (T3 - T4) / R_loss(fan level)
    z    <- clip(z - (q_in - q_out) * dt / Euse, 0, 1)
with the real dt, starting from the run's recorded initial PCM condition, and
re-anchored to 1 or 0 when T1-T3 meet the regenerated or depleted endpoint rule.
Every re-anchoring jump is recorded: it is E0's drift measurement.
All constants come from the calibration package; nothing is assumed here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import Config

INITIAL_STATE_Z = {"regenerated": 1.0, "depleted": 0.0}   # Part III: z = 1 regenerated, 0 depleted


@dataclass
class Endpoints:
    regenerated_below_c: float
    regenerated_hold_s: float
    depleted_above_c: float
    depleted_hold_s: float

    @classmethod
    def from_config(cls, cfg: Config) -> "Endpoints":
        return cls(*(float(cfg.require(f"estimator.endpoints.{k}")) for k in
                     ("regenerated_below_c", "regenerated_hold_s", "depleted_above_c", "depleted_hold_s")))


class EndpointDetector:
    """Tracks how long T1-T3 have all been beyond each endpoint threshold."""

    def __init__(self, ep: Endpoints):
        self.ep = ep
        self.below_s = 0.0
        self.above_s = 0.0
        self.state = None

    def update(self, t1, t2, t3, dt) -> str | None:
        temps = np.array([t1, t2, t3], float)
        if not np.all(np.isfinite(temps)):
            self.below_s = self.above_s = 0.0
            return None
        self.below_s = self.below_s + dt if np.all(temps < self.ep.regenerated_below_c) else 0.0
        self.above_s = self.above_s + dt if np.all(temps > self.ep.depleted_above_c) else 0.0
        now = ("regenerated" if self.below_s >= self.ep.regenerated_hold_s else
               "depleted" if self.above_s >= self.ep.depleted_hold_s else None)
        reached = now if now is not None and now != self.state else None
        self.state = now
        return reached          # reported once, when the endpoint is first reached


@dataclass
class E0Estimator:
    euse_j: float
    r_proc_k_per_w: float
    r_loss_by_fan: dict          # fan_command_pct -> K/W
    endpoints: Endpoints
    z: float = float("nan")
    lost: bool = True
    reanchors: list = field(default_factory=list)

    @classmethod
    def from_reference(cls, ref: pd.DataFrame, endpoints: Endpoints) -> "E0Estimator":
        """ref: load_reference() output for one cartridge, indexed by fan level."""
        for col in ("euse_mean_j", "r_proc_k_per_w"):
            if ref[col].nunique() != 1:
                raise ValueError(f"{col} must have one value per cartridge in the calibration package")
        return cls(euse_j=float(ref["euse_mean_j"].iloc[0]), r_proc_k_per_w=float(ref["r_proc_k_per_w"].iloc[0]),
                   r_loss_by_fan={float(k): float(v) for k, v in ref["r_loss_k_per_w"].items()},
                   endpoints=endpoints)

    def start_run(self, initial_pcm_state: str):
        if initial_pcm_state not in INITIAL_STATE_Z:
            raise ValueError(f"E0 needs a known starting point; initial_pcm_state '{initial_pcm_state}' "
                             f"is not one of {list(INITIAL_STATE_Z)}")
        self.z = INITIAL_STATE_Z[initial_pcm_state]
        self.lost = False
        self.reanchors = []
        self._det = EndpointDetector(self.endpoints)

    def step(self, elapsed_s, dt, soc, t1, t2, t3, t4, fan) -> float:
        """Returns z, or NaN when an input is missing (the estimate then stays invalid until
        the next endpoint re-anchors it)."""
        endpoint = self._det.update(t1, t2, t3, dt)
        r_loss = self.r_loss_by_fan.get(float(fan)) if np.isfinite(fan) else None
        inputs_ok = all(np.isfinite([soc, t1, t3, t4, dt])) and r_loss is not None
        if inputs_ok and not self.lost:
            q_in = (soc - t1) / self.r_proc_k_per_w
            q_out = (t3 - t4) / r_loss
            self.z = float(np.clip(self.z - (q_in - q_out) * dt / self.euse_j, 0.0, 1.0))
        elif not inputs_ok:
            self.lost = True
        if endpoint is not None:
            target = INITIAL_STATE_Z[endpoint]
            self.reanchors.append({"elapsed_s": elapsed_s, "endpoint": endpoint,
                                   "z_before": None if self.lost else self.z,
                                   "jump": None if self.lost else target - self.z})
            self.z, self.lost = target, False
        return float("nan") if self.lost else self.z

    def run(self, df: pd.DataFrame, initial_pcm_state: str, fan_field: str = "fan_command_pct") -> np.ndarray:
        self.start_run(initial_pcm_state)
        el = df["elapsed_s"].to_numpy(float)
        dt = np.diff(el, prepend=el[0])
        cols = [df[c].to_numpy(float) for c in ("soc_temp_c", "pcm_t1_c", "pcm_t2_c", "pcm_t3_c",
                                                 "ambient_temp_c", fan_field)]
        return np.array([self.step(el[i], dt[i], *(c[i] for c in cols)) for i in range(len(df))])
