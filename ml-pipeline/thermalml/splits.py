"""Run-level split manifest (Part III 4.3.3.1 B/C, NFR6).

Which run goes to train, validation or test is a team decision recorded in the
manifest file; this module never assigns splits itself. It only builds an empty
template, checks a filled manifest, hashes it for model_metadata.json, and gives
leave-one-run-out folds for the optional stability check.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

SPLITS = ("train", "validation", "test")
RUN_MANIFEST_COLUMNS = ["run_id", "split", "control_mode", "chamber_setpoint_c", "workload_config",
                        "schedule_id", "initial_pcm_state", "cartridge_id"]
CYCLE_MANIFEST_COLUMNS = ["cycle_id", "split", "fan_command_pct", "cartridge_id"]


class ManifestError(ValueError):
    pass


def make_run_manifest_template(run_configs_dir: Path, out: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(Path(run_configs_dir).glob("*.json")):
        with open(p) as f:
            c = json.load(f)
        rows.append({k: (c.get(k, "") if k != "split" else "") for k in RUN_MANIFEST_COLUMNS})
    df = pd.DataFrame(rows, columns=RUN_MANIFEST_COLUMNS)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def load_manifest(path: Path, id_col: str = "run_id") -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if id_col not in df.columns or "split" not in df.columns:
        raise ManifestError(f"{path}: needs '{id_col}' and 'split' columns")
    if df[id_col].duplicated().any():
        dup = df.loc[df[id_col].duplicated(), id_col].tolist()
        raise ManifestError(f"{path}: {id_col} listed more than once: {dup}")
    blank = df.loc[df["split"].str.strip() == "", id_col].tolist()
    if blank:
        raise ManifestError(f"{path}: split not assigned for {blank}")
    bad = sorted(set(df["split"]) - set(SPLITS))
    if bad:
        raise ManifestError(f"{path}: unknown split names {bad}; use {SPLITS}")
    for s in ("train", "validation"):
        if not (df["split"] == s).any():
            raise ManifestError(f"{path}: no '{s}' entries")
    return df


def ids(manifest: pd.DataFrame, split: str, id_col: str = "run_id") -> list[str]:
    return manifest.loc[manifest["split"] == split, id_col].tolist()


def manifest_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def leave_one_out(manifest: pd.DataFrame, id_col: str = "run_id"):
    """Folds over train+validation runs only; test runs are never used here."""
    pool = manifest.loc[manifest["split"].isin(["train", "validation"]), id_col].tolist()
    for held in pool:
        yield [r for r in pool if r != held], held


class TestLock:
    """Part III 4.3.3.1 C: test runs are used once, after model selection is complete.

    Scoring the test split writes a lock file; a second scoring is refused unless
    the caller explicitly overrides it, and the override is recorded in the lock.
    """

    def __init__(self, results_dir: Path, name: str):
        self.path = Path(results_dir) / f"TEST_SCORED_{name}.json"

    def check_and_set(self, info: dict, override: bool = False):
        if self.path.exists() and not override:
            raise RuntimeError(
                f"Test split already scored ({self.path}). Part III uses test runs once. "
                "Pass --override-test-lock only if the team agrees, and report it.")
        history = []
        if self.path.exists():
            history = json.loads(self.path.read_text()).get("history", [])
        history.append(info)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"history": history}, indent=2, default=str))
