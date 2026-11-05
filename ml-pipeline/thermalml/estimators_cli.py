"""PCM capacity estimators E0-E2 (Part III 4.3.3.1 A).

    python -m thermalml.estimators_cli reference --cartridge C1   # z_ref for every heater cycle, Euse per melt cycle
    python -m thermalml.estimators_cli train     --cartridge C1   # E1 and E2 on train/validation cycles
    python -m thermalml.estimators_cli test      --cartridge C1   # once: per-cycle errors, E2-vs-E1 rule
    python -m thermalml.estimators_cli e0-drift  --cartridge C1   # E0 re-anchoring jumps on operational runs
    python -m thermalml.estimators_cli apply     --estimator e0|<package> [--reset-at-endpoints]
                                                                  # z_hat_offline for every run (predictor M4 input)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics, splits
from .config import Config
from .estimators import calibration as cal
from .estimators import learned
from .estimators.e0 import E0Estimator, EndpointDetector, Endpoints
from .estimators.features import estimator_features
from .features import Z_COLUMN, hold, regularize
from .io import load_run, load_run_config
from .package import read_package, write_package


def _cal(cfg, name):
    return cfg.path_of("calibration_dir") / name


def load_cycles(cfg: Config) -> dict[str, pd.DataFrame]:
    out = {}
    for p in sorted(cfg.path_of("calibration_dir").glob("calibration_*.csv")):
        c = cal.load_cycle(p)
        out[str(c["cycle_id"].iloc[0])] = c
    if not out:
        raise FileNotFoundError(f"no calibration_*.csv files in {cfg.path_of('calibration_dir')}")
    return out


def cycle_matrix(cycle: pd.DataFrame, cfg: Config, z_ref: np.ndarray | None):
    """Features on the control-period grid; leading rows without full slope history are dropped
    so E1 and E2 are scored on identical rows."""
    period = float(cfg.require("timing.sampling_period_s"))
    k = np.rint(cycle["elapsed_s"].to_numpy() / period)
    if len(np.unique(k)) != len(k) or np.any(np.diff(k) != 1):
        raise ValueError(f"cycle {cycle['cycle_id'].iloc[0]}: samples are not one per control period without gaps")
    X = estimator_features(cycle, cfg)
    first = int(np.argmax(np.isfinite(X.to_numpy()).all(axis=1)))
    if not np.isfinite(X.to_numpy()[first:]).all():
        raise ValueError(f"cycle {cycle['cycle_id'].iloc[0]}: missing values inside the cycle")
    return X.iloc[first:].reset_index(drop=True), (None if z_ref is None else z_ref[first:])


def reference(cfg: Config, cartridge: str):
    ref = cal.load_reference(_cal(cfg, "pcm_capacity_reference.csv"), cartridge)
    rig = cal.load_rig_loss(_cal(cfg, "rig_loss.csv"))
    c_fix = float(ref["c_fix_j_per_k"].iloc[0])
    euse = float(ref["euse_mean_j"].iloc[0])
    out_dir = _cal(cfg, "reference_z")
    out_dir.mkdir(parents=True, exist_ok=True)
    euse_rows = []
    for cid, c in load_cycles(cfg).items():
        z = cal.reference_z(c, rig, c_fix, euse)
        pd.DataFrame({"cycle_id": cid, "elapsed_s": c["elapsed_s"], "z_ref": z}).to_csv(out_dir / f"{cid}.csv", index=False)
        if c["phase"].iloc[0] == "melt":
            euse_rows.append({"cycle_id": cid, "fan_command_pct": c["fan_command_pct"].iloc[0],
                              "euse_j": cal.euse_from_melt_cycle(c, rig, c_fix)})
        else:
            euse_rows.append({"cycle_id": cid, "fan_command_pct": c["fan_command_pct"].iloc[0],
                              "heat_removed_j": float(cal.cycle_energy(c, rig, c_fix)[-1])})
    pd.DataFrame(euse_rows).to_csv(_cal(cfg, "cycle_energy_summary.csv"), index=False)
    print(f"reference z in {out_dir}; per-cycle energies in {_cal(cfg, 'cycle_energy_summary.csv')}")


def _load_ref_z(cfg, cid):
    return pd.read_csv(_cal(cfg, "reference_z") / f"{cid}.csv")["z_ref"].to_numpy()


def _per_cycle_rows(name, split, cids, preds, refs):
    rows = []
    for cid, p, z in zip(cids, preds, refs):
        err = (p - z) * 100.0           # percentage points of z (Part III 4.3.3.1 C)
        rows.append({"estimator": name, "split": split, "cycle_id": cid, "n": len(err),
                     "mae_pp": metrics.mae(err), "rmse_pp": metrics.rmse(err)})
    return rows


def train(cfg: Config, cartridge: str):
    man = splits.load_manifest(cfg.path_of("cycle_split_manifest"), id_col="cycle_id")
    cycles = load_cycles(cfg)
    data = {cid: cycle_matrix(cycles[cid], cfg, _load_ref_z(cfg, cid)) for cid in man["cycle_id"]}
    tr, va = splits.ids(man, "train", "cycle_id"), splits.ids(man, "validation", "cycle_id")
    Xtr = pd.concat([data[c][0] for c in tr]); ytr = np.concatenate([data[c][1] for c in tr])
    Xva = pd.concat([data[c][0] for c in va]); yva = np.concatenate([data[c][1] for c in va])
    rows, best = [], {}
    kind = cfg.require("estimator.e1.model")
    for st in learned.grid_settings(cfg.require("estimator.e1.grid")):
        m = learned.fit_e1(kind, Xtr, ytr, Xva, yva, st, cfg.get("estimator.e1.early_stopping_rounds"))
        r = _per_cycle_rows("E1", "validation", va, [learned.predict_e1(m, data[c][0]) for c in va],
                            [data[c][1] for c in va])
        for x in r:
            x["setting"] = json.dumps(st, sort_keys=True)
        rows += r
        score = np.mean([x["rmse_pp"] for x in r])
        if "E1" not in best or score < best["E1"][1]:
            best["E1"] = (m, score, st)
    e2c = cfg.require("estimator.e2")
    period = float(cfg.require("timing.sampling_period_s"))
    chunk = int(round(float(cfg.require("estimator.e2.chunk_s")) / period))
    for st in learned.grid_settings(cfg.require("estimator.e2.grid")):
        m, info = learned.train_e2(e2c["cell"], [(data[c][0].to_numpy(), data[c][1]) for c in tr],
                                   [(data[c][0].to_numpy(), data[c][1]) for c in va], int(st["hidden_units"]), chunk,
                                   float(cfg.require("estimator.e2.learning_rate")),
                                   int(cfg.require("estimator.e2.max_epochs")),
                                   int(cfg.require("estimator.e2.early_stopping_patience")),
                                   int(cfg.require("estimator.e2.seed")))
        r = _per_cycle_rows("E2", "validation", va, [learned.predict_e2(m, data[c][0].to_numpy()) for c in va],
                            [data[c][1] for c in va])
        for x in r:
            x["setting"] = json.dumps(st, sort_keys=True)
        rows += r
        score = np.mean([x["rmse_pp"] for x in r])
        if "E2" not in best or score < best["E2"][1]:
            best["E2"] = (m, score, st, info)
    res = cfg.path_of("results_dir")
    res.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(res / "estimator_validation.csv", index=False)
    common = dict(kind="pcm_estimator", cartridge_id=cartridge, feature_list=list(Xtr.columns),
                  training_cycle_ids=tr, validation_cycle_ids=va,
                  split_manifest_sha256=splits.manifest_hash(cfg.path_of("cycle_split_manifest")),
                  sampling_period_s=period, slope_windows_s=cfg.require("estimator.slope_windows_s"),
                  transition_range_c=cfg.require("estimator.transition_range_c"))
    m1, s1, st1 = best["E1"]
    folder = cfg.path_of("models_dir") / "estimator_E1"
    folder.mkdir(parents=True, exist_ok=True)
    if kind == "xgboost":
        m1.save_model(str(folder / "model.json")); f1 = "model.json"
    else:
        import joblib
        joblib.dump(m1, folder / "model.joblib"); f1 = "model.joblib"
    write_package(cfg.path_of("models_dir"), "estimator_E1",
                  dict(common, model="E1", e1_kind=kind, model_file=f1, setting=st1, val_rmse_pp_mean=s1))
    m2, s2, st2, info2 = best["E2"]
    folder = cfg.path_of("models_dir") / "estimator_E2"
    learned.export_e2_onnx(m2, Xtr.shape[1], folder / "model.onnx")
    check = _e2_onnx_check(m2, folder / "model.onnx", data[va[0]][0].to_numpy())
    write_package(cfg.path_of("models_dir"), "estimator_E2",
                  dict(common, model="E2", cell=e2c["cell"], model_file="model.onnx", setting=st2, training_info=info2,
                       val_rmse_pp_mean=s2, hidden_units=int(st2["hidden_units"]), onnx_check=check,
                       scaler={"mean": m2.x_mean.tolist(), "std": m2.x_std.tolist()}))
    print(f"estimator packages in {cfg.path_of('models_dir')}; validation in {res/'estimator_validation.csv'}")


def _e2_onnx_check(model, path, x):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    torch_z = learned.predict_e2(model, x)
    h = np.zeros((1, 1, model.hidden), np.float32)
    c = np.zeros_like(h)
    out = []
    for row in x:
        feeds = {"x": row[None, None].astype(np.float32), "h_in": h}
        if model.cell_type == "lstm":
            feeds["c_in"] = c
            z, h, c = sess.run(None, feeds)
        else:
            z, h = sess.run(None, feeds)
        out.append(float(np.ravel(z)[0]))
    return {"max_abs_diff_z": float(np.max(np.abs(np.array(out) - torch_z))), "steps": len(out)}


def _load_estimator(cfg, name):
    folder = cfg.path_of("models_dir") / name
    meta = read_package(folder)
    if meta["model"] == "E1":
        if meta["e1_kind"] == "xgboost":
            import xgboost as xgb
            m = xgb.XGBRegressor(n_jobs=1); m.load_model(str(folder / meta["model_file"]))
        else:
            import joblib
            m = joblib.load(folder / meta["model_file"])
        return meta, lambda X, resets=None: learned.predict_e1(m, X[meta["feature_list"]])
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(folder / meta["model_file"]), so, providers=["CPUExecutionProvider"])

    def run(X, resets=None):
        x = X[meta["feature_list"]].to_numpy(np.float32)
        resets = set() if resets is None else set(map(int, resets))
        h = np.zeros((1, 1, meta["hidden_units"]), np.float32); c = np.zeros_like(h)
        out = np.full(len(x), np.nan)
        for i, row in enumerate(x):
            if i in resets:
                h = np.zeros_like(h); c = np.zeros_like(c)
            if not np.all(np.isfinite(row)):
                continue
            feeds = {"x": row[None, None], "h_in": h}
            if meta["cell"] == "lstm":
                feeds["c_in"] = c
                z, h, c = sess.run(None, feeds)
            else:
                z, h = sess.run(None, feeds)
            out[i] = float(np.ravel(z)[0])
        return out
    return meta, run


def test(cfg: Config, override_lock: bool):
    man = splits.load_manifest(cfg.path_of("cycle_split_manifest"), id_col="cycle_id")
    te = splits.ids(man, "test", "cycle_id")
    if not te:
        raise ValueError("no test cycles in the cycle split manifest")
    splits.TestLock(cfg.path_of("results_dir"), "estimators").check_and_set({"test_cycles": te}, override_lock)
    cycles = load_cycles(cfg)
    data = {c: cycle_matrix(cycles[c], cfg, _load_ref_z(cfg, c)) for c in te}
    rows = []
    for name in ("estimator_E1", "estimator_E2"):
        meta, fn = _load_estimator(cfg, name)
        rows += _per_cycle_rows(meta["model"], "test", te, [fn(data[c][0]) for c in te], [data[c][1] for c in te])
    df = pd.DataFrame(rows)
    out = cfg.path_of("results_dir")
    df.to_csv(out / "estimator_test.csv", index=False)
    piv = df.pivot_table(index="cycle_id", columns="estimator", values="rmse_pp")
    d = (piv["E1"] - piv["E2"]).to_numpy()       # positive = E2 better
    rule = {"cycles": len(d), "mean_improvement_pp": float(d.mean()),
            "spread_std_pp": float(np.std(d, ddof=1)) if len(d) > 1 else float("nan"),
            "rule": "Part III 4.3.3.1 A: keep E2 only if its improvement on E1 is larger than the per-cycle spread"}
    rule["keep_E2"] = bool(len(d) > 1 and rule["mean_improvement_pp"] > rule["spread_std_pp"])
    (out / "estimator_E2_vs_E1.json").write_text(json.dumps(rule, indent=2))
    print(json.dumps(rule, indent=2))


def _run_frames(cfg, ids):
    for rid in ids:
        df = load_run(cfg.path_of("runs_dir") / f"{rid}.csv")
        rc = load_run_config(cfg.path_of("run_configs_dir") / f"{rid}.json")
        yield rid, df, rc


def _e0(cfg, cartridge):
    ref = cal.load_reference(_cal(cfg, "pcm_capacity_reference.csv"), cartridge)
    return E0Estimator.from_reference(ref, Endpoints.from_config(cfg))


def e0_drift(cfg: Config, cartridge: str):
    man = splits.load_manifest(cfg.path_of("split_manifest"))
    rows = []
    for rid, df, rc in _run_frames(cfg, man["run_id"].tolist()):
        if rc["control_mode"] == "M1":
            continue                     # no cartridge in M1
        est = _e0(cfg, rc["cartridge_id"] if cartridge is None else cartridge)
        est.run(df, rc["initial_pcm_state"])
        rows += [dict(run_id=rid, **r) for r in est.reanchors]
    pd.DataFrame(rows).to_csv(cfg.path_of("results_dir") / "e0_drift.csv", index=False)


def apply(cfg: Config, estimator: str, reset_at_endpoints: bool, cartridge: str | None):
    """Writes results/z_offline/<run_id>.csv with the selected estimator's output for every run
    (Part III 4.3.3.1 B: the PCM-aware predictor trains on the estimator output it will see online)."""
    man = splits.load_manifest(cfg.path_of("split_manifest"))
    out_dir = cfg.path_of("results_dir") / "z_offline"
    out_dir.mkdir(parents=True, exist_ok=True)
    period = float(cfg.require("timing.sampling_period_s"))
    hold_n = int(cfg.require("missing_data.hold_max_periods"))
    for rid, df, rc in _run_frames(cfg, man["run_id"].tolist()):
        if rc["control_mode"] == "M1":
            continue
        if estimator == "e0":
            est = _e0(cfg, cartridge or rc["cartridge_id"])
            z = est.run(df, rc["initial_pcm_state"]); version = "E0 (calibration package)"
            el = df["elapsed_s"].to_numpy()
        else:
            meta, fn = _load_estimator(cfg, estimator)
            g = regularize(df, period)
            fields = ["pcm_t1_c", "pcm_t2_c", "pcm_t3_c", cfg.require("estimator.ambient_field"),
                      cfg.require("estimator.fan_field")]
            held, _ = hold(g, fields, hold_n)
            X = estimator_features(held, cfg)
            resets = None
            if reset_at_endpoints:
                det, resets = EndpointDetector(Endpoints.from_config(cfg)), []
                for i, (a, b, c) in enumerate(held[["pcm_t1_c", "pcm_t2_c", "pcm_t3_c"]].to_numpy()):
                    if det.update(a, b, c, period):
                        resets.append(i)
            z = fn(X, resets); version = meta["version"]
            el = g["elapsed_s"].to_numpy()
        pd.DataFrame({"elapsed_s": el, Z_COLUMN: z, "estimator_version": version,
                      "reset_at_endpoints": reset_at_endpoints}).dropna(subset=["elapsed_s"]).to_csv(
            out_dir / f"{rid}.csv", index=False)
    print(f"z_hat_offline written to {out_dir}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["reference", "train", "test", "e0-drift", "apply"])
    ap.add_argument("--config", default="config/pipeline_config.yaml")
    ap.add_argument("--cartridge")
    ap.add_argument("--estimator", help="'e0', 'estimator_E1' or 'estimator_E2' (apply)")
    ap.add_argument("--reset-at-endpoints", action="store_true")
    ap.add_argument("--override-test-lock", action="store_true")
    a = ap.parse_args(argv)
    cfg = Config.load(a.config)
    cfg.path_of("results_dir").mkdir(parents=True, exist_ok=True)
    if a.command in ("reference", "train") and not a.cartridge:
        ap.error("--cartridge is required")
    {"reference": lambda: reference(cfg, a.cartridge), "train": lambda: train(cfg, a.cartridge),
     "test": lambda: test(cfg, a.override_test_lock), "e0-drift": lambda: e0_drift(cfg, a.cartridge),
     "apply": lambda: apply(cfg, a.estimator, a.reset_at_endpoints, a.cartridge)}[a.command]()


if __name__ == "__main__":
    main()
