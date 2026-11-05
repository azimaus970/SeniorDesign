"""Load the pipeline configuration and refuse to run steps whose values are not fixed yet.

No numeric default is invented here: a value is either in the YAML file (with its
source noted there) or the step that needs it stops with a clear message.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml


class ConfigMissing(RuntimeError):
    """A value the step needs is still null in the config file."""


class Config:
    def __init__(self, data: dict, path: Path | None = None, root: Path | None = None):
        self.data = data
        self.path = path
        self.root = root or (path.parent.parent if path else Path.cwd())

    @classmethod
    def load(cls, path: str | Path, root: str | Path | None = None) -> "Config":
        p = Path(path)
        with open(p) as f:
            data = yaml.safe_load(f)
        return cls(data, p, Path(root) if root else None)

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, dotted: str) -> Any:
        value = self.get(dotted)
        if value is None:
            raise ConfigMissing(
                f"'{dotted}' is not set in {self.path or 'config'}. It is a team decision or a "
                "value from pilot data; set it before running this step."
            )
        return value

    def path_of(self, key: str) -> Path:
        p = Path(self.require(f"paths.{key}"))
        return p if p.is_absolute() else self.root / p


# Values each pipeline step needs. check_config reports which are still null.
REQUIRED_BY_STEP = {
    "load logs": ["timing.sampling_period_s"],
    "build windows (missing data)": ["missing_data.hold_max_periods", "missing_data.max_missing_fraction"],
    "predictor targets": ["predictor.horizons_s", "predictor.event_kinds_that_break_target"],
    "XGBoost predictor features": ["predictor.lags_s", "predictor.slope_windows_s"],
    "linear-trend baseline": ["baselines.linear_trend_window_s"],
    "XGBoost training": ["xgboost.grid", "xgboost.early_stopping_rounds"],
    "GRU/LSTM training": ["rnn.grid", "rnn.learning_rate", "rnn.batch_size", "rnn.max_epochs",
                          "rnn.early_stopping_patience", "rnn.seed"],
    "model selection": ["predictor.primary_horizon_s"],
    "paired variant comparison": ["evaluation.bootstrap_resamples", "evaluation.confidence_level",
                                  "evaluation.seed"],
    "E0 and reference z": ["estimator.endpoints.regenerated_below_c", "estimator.endpoints.regenerated_hold_s",
                           "estimator.endpoints.depleted_above_c", "estimator.endpoints.depleted_hold_s"],
    "E1/E2 features": ["estimator.slope_windows_s", "estimator.transition_range_c"],
    "E1 training": ["estimator.e1.grid"],
    "E2 training": ["estimator.e2.grid", "estimator.e2.chunk_s", "estimator.e2.learning_rate",
                    "estimator.e2.max_epochs", "estimator.e2.early_stopping_patience", "estimator.e2.seed"],
    "on-Pi timing check": ["runtime_limits.limit_statistic", "runtime_limits.max_call_ms",
                           "runtime_limits.max_combined_ms"],
}


def check(cfg: Config) -> dict[str, list[str]]:
    missing = {}
    for step, keys in REQUIRED_BY_STEP.items():
        m = [k for k in keys if cfg.get(k) is None]
        if m:
            missing[step] = m
    # Range rules that Part III does fix
    grid = cfg.get("rnn.grid") or {}
    lo, hi = cfg.get("rnn.hidden_units_allowed_range", [None, None])
    for h in grid.get("hidden_units", []) or []:
        if lo is not None and not (lo <= h <= hi):
            missing.setdefault("GRU/LSTM training", []).append(
                f"rnn.grid.hidden_units value {h} is outside Part III's {lo}-{hi} range")
    for n in grid.get("layers", []) or []:
        if n not in cfg.get("rnn.layers_allowed", []):
            missing.setdefault("GRU/LSTM training", []).append(
                f"rnn.grid.layers value {n} is not allowed by Part III")
    return missing


def main(argv=None):
    argv = argv or sys.argv[1:]
    cfg = Config.load(argv[0] if argv else "config/pipeline_config.yaml")
    missing = check(cfg)
    if not missing:
        print("All required values are set.")
        return 0
    print("Values still to be fixed by the team (null in the config):")
    for step, keys in missing.items():
        print(f"  {step}:")
        for k in keys:
            print(f"    - {k}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
