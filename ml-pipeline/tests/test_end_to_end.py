"""Runs every CLI step once on tiny hand-built files in a temporary folder, to prove the
steps connect. See tests/helpers.py: these values are not data and produce no results
for the project; everything is deleted after the test.
"""
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from helpers import frame, make_test_config, write_run
from thermalml import estimators_cli, predictors
from thermalml.config import Config

EST_FILL = {
    "estimator.endpoints.regenerated_below_c": 2.0, "estimator.endpoints.regenerated_hold_s": 2.0,
    "estimator.endpoints.depleted_above_c": 8.0, "estimator.endpoints.depleted_hold_s": 2.0,
    "estimator.slope_windows_s": [3], "estimator.transition_range_c": [4.0, 6.0],
    "estimator.e1.grid": {"max_depth": [2], "n_estimators": [10]},
    "estimator.e2.grid": {"hidden_units": [4]}, "estimator.e2.chunk_s": 20,
    "estimator.e2.learning_rate": 0.01, "estimator.e2.max_epochs": 2,
    "estimator.e2.early_stopping_patience": 1, "estimator.e2.seed": 0,
}


def _setup(tmp):
    cfg = make_test_config(tmp, **EST_FILL)
    (tmp / "config").mkdir()
    cfg_path = tmp / "config" / "c.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg.data))
    n = 300
    t = np.arange(n, dtype=float)
    ids = {"R1": "train", "R2": "train", "R3": "validation", "R4": "test", "R5": "test"}
    for i, rid in enumerate(ids):
        pcm = 1.0 + 9.0 * t / n
        df = frame(n, rid, temp=np.sin(t / (7 + i)) + t * 0.01 * (i + 1), pcm_t1_c=pcm, pcm_t2_c=pcm,
                   pcm_t3_c=pcm, cpu_usage_pct=np.cos(t / 5))
        write_run(df, tmp / "data/runs")
        (tmp / "data/run_configs").mkdir(parents=True, exist_ok=True)
        (tmp / "data/run_configs" / f"{rid}.json").write_text(json.dumps(dict(
            run_id=rid, control_mode="M2", chamber_setpoint_c="x", sampling_period_s=1,
            initial_pcm_state="regenerated", cartridge_id="C", workload_config="w", schedule_id="s")))
    (tmp / "datasets").mkdir()
    pd.DataFrame({"run_id": list(ids), "split": list(ids.values())}).to_csv(tmp / "datasets/split_manifest.csv", index=False)
    cal = tmp / "calibration"
    cal.mkdir()
    pd.DataFrame([dict(cartridge_id="C", fan_command_pct=1.0, euse_mean_j=500.0, euse_std_j=1.0, uncertainty_j=1.0,
                       c_fix_j_per_k=1.0, r_loss_k_per_w=1.0, r_proc_k_per_w=1.0)]).to_csv(
        cal / "pcm_capacity_reference.csv", index=False)
    pd.DataFrame({"fan_command_pct": [1.0, 1.0], "delta_t_fix_amb_k": [-50.0, 50.0],
                  "p_loss_w": [0.0, 1.0]}).to_csv(cal / "rig_loss.csv", index=False)
    cyc = {"K1": "train", "K2": "train", "K3": "validation", "K4": "test", "K5": "test"}
    for j, cid in enumerate(cyc):
        m = 80
        x = np.arange(m, dtype=float)
        temp = 1.0 + 9.0 * x / m + 0.1 * j
        pd.DataFrame({"cycle_id": cid, "elapsed_s": x, "phase": "melt", "heater_voltage_v": 7.0 + j,
                      "heater_current_a": 1.0, "fixture_temp_c": temp + 1, "pcm_t1_c": temp, "pcm_t2_c": temp,
                      "pcm_t3_c": temp - 0.5, "ambient_temp_c": 1.0, "fan_command_pct": 1.0}).to_csv(
            cal / f"calibration_{j:03d}.csv", index=False)
    pd.DataFrame({"cycle_id": list(cyc), "split": list(cyc.values())}).to_csv(
        tmp / "datasets/cycle_split_manifest.csv", index=False)
    return str(cfg_path)


def test_full_pipeline(tmp_path):
    c = _setup(tmp_path)
    estimators_cli.main(["reference", "--config", c, "--cartridge", "C"])
    estimators_cli.main(["train", "--config", c, "--cartridge", "C"])
    estimators_cli.main(["test", "--config", c])
    with pytest.raises(RuntimeError):
        estimators_cli.main(["test", "--config", c])            # test split is used once
    estimators_cli.main(["e0-drift", "--config", c, "--cartridge", "C"])
    estimators_cli.main(["apply", "--config", c, "--estimator", "estimator_E2", "--reset-at-endpoints"])
    predictors.main(["train", "--config", c])
    predictors.main(["test", "--config", c])
    predictors.main(["loro", "--config", c, ])
    res = tmp_path / "results"
    ev = pd.read_csv(res / "evaluation_metrics.csv")
    assert {"persistence", "linear_trend", "xgboost", "gru", "lstm"} <= set(ev["model"])
    assert set(ev["split"]) == {"validation", "test"}
    sel = pd.read_csv(res / "selection.csv")
    assert len(sel) == 6                                       # 3 models x 2 variants at the primary horizon
    assert (sel.groupby("model")["window_s"].nunique() == 1).all()   # M3 and M4 share the window (Part III)
    for pkg in sel["package"]:
        meta = json.loads((tmp_path / "models" / pkg / "metadata.json").read_text())
        assert meta["split_manifest_sha256"] and meta["feature_list"]
        if "onnx_check" in meta:
            assert meta["onnx_check"]["max_abs_diff_c"] < 1e-4
    e2 = json.loads((tmp_path / "models/estimator_E2/metadata.json").read_text())
    assert e2["onnx_check"]["max_abs_diff_z"] < 1e-4
    assert (res / "variant_comparison.csv").exists() and (res / "loro_stability.csv").exists()


def test_bench_runs(tmp_path, capsys):
    c = _setup(tmp_path)
    predictors.main(["train", "--config", c, "--variants", "temperature_only", "--models", "xgboost", "gru"])
    from thermalml import bench
    bench.main(["--config", c, "--run", str(tmp_path / "data/runs/R4.csv"), "--calls", "5", "--packages",
                "predictor_xgboost_temperature_only_H10", "predictor_gru_temperature_only_H10"])
    assert "p95_ms" in capsys.readouterr().out
