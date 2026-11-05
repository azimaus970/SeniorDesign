"""
Functional tests for edge_workload.py. They run the real model on the real
image subset, so they take about 2 minutes.

    DATA=/path/to/imagenette2-320/val pytest -q tests/

The calibration fraction used here (0.5) is a TEST value only, to exercise the
reduced_load code path. The real fraction is a team decision made on the Pi.
"""
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(HERE, "edge_workload.py")
DATA = os.environ.get("DATA")
MODELS = os.environ.get("MODELS", os.path.join(HERE, "models"))
pytestmark = pytest.mark.skipif(not DATA, reason="set DATA to the Imagenette val folder")


def run(*args, check=True, **kw):
    cmd = [sys.executable, SCRIPT, *args]
    p = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if check and p.returncode != 0:
        raise AssertionError(p.stdout + p.stderr)
    return p


def rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


@pytest.fixture()
def tmp_profiles(tmp_path):
    dst = tmp_path / "profiles.json"
    shutil.copy(os.path.join(HERE, "profiles.json"), dst)
    return str(dst)


def common(profiles):
    return ["--data", DATA, "--models-dir", MODELS, "--profiles", profiles]


def test_reduced_load_refused_until_calibrated(tmp_path, tmp_profiles):
    p = run("run", *common(tmp_profiles), "--profile", "reduced_load", "--duration", "2",
            "--out-dir", str(tmp_path), check=False)
    assert p.returncode != 0 and "no measured rate" in (p.stdout + p.stderr)


def test_schedule_control_file_and_log_format(tmp_path, tmp_profiles):
    # calibrate with a TEST fraction so reduced_load can be exercised
    run("calibrate", *common(tmp_profiles), "--seconds", "5", "--fraction", "0.5", "--write")
    rate = json.load(open(tmp_profiles))["profiles"]["reduced_load"]["rate_hz"]
    assert rate and rate > 0

    sched = tmp_path / "sched.json"
    sched.write_text(json.dumps([{"t": 0, "profile": "high_performance"}, {"t": 6, "profile": "balanced"}]))
    ctrl, qos = tmp_path / "control.json", tmp_path / "qos.json"
    proc = subprocess.Popen([sys.executable, SCRIPT, "run", *common(tmp_profiles), "--duration", "22",
                             "--run-id", "TEST1", "--pcm-init", "test", "--schedule", str(sched),
                             "--control", str(ctrl), "--qos-file", str(qos), "--out-dir", str(tmp_path)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    time.sleep(14)
    ctrl.write_text("{not json")                       # must be logged, not crash
    time.sleep(2)
    run("set-profile", "reduced_load", "--control", str(ctrl))
    out, _ = proc.communicate(timeout=60)
    assert proc.returncode == 0, out

    r = rows(tmp_path / "TEST1_ai.csv")
    ev = rows(tmp_path / "TEST1_events.csv")
    profiles_seen = [x["workload_profile"] for x in r]
    assert profiles_seen[0] == "high_performance"
    assert "balanced" in profiles_seen and profiles_seen[-1] == "reduced_load"
    assert any(e["event"] == "control_error" for e in ev)
    # each row holds one profile; transitions flagged; change time recorded
    for x in r:
        assert x["profile_changed_t_unix"] != ""
    flagged = [i for i, x in enumerate(r) if x["transition"] == "1"]
    changes = [i for i in range(1, len(r)) if r[i]["workload_profile"] != r[i - 1]["workload_profile"]]
    assert set(changes) <= set(flagged)
    # windows end on whole seconds (after the first partial one)
    for x in r[1:]:
        frac = float(x["t_unix"]) % 1
        assert frac < 0.25 or frac > 0.99, x["t_unix"]
    # missing values are empty, never "0"
    for x in r:
        if x["n_inf"] == "0":
            assert x["inference_latency_ms"] == ""
    # exact correct counts are logged as integers alongside the completed counts
    for x in r:
        assert 0 <= int(x["n_correct"]) <= int(x["n_inf"])
    # reduced_load respects its rate cap (allow 15% for 1 s windows)
    steady = [x for x in r if x["workload_profile"] == "reduced_load" and x["transition"] == "0"]
    assert steady, "no steady reduced_load windows"
    for x in steady:
        assert float(x["inference_fps"]) <= rate * 1.15 + 1
    # each row is published after its window ends, and names the request in force
    for x in r:
        assert float(x["published_t_unix"]) >= float(x["t_unix"])
        assert x["request_t_unix"] != ""
    last = [x for x in r if x["workload_profile"] == "reduced_load"][-1]
    assert abs(float(last["request_t_unix"]) - json.load(open(ctrl))["requested_t_unix"]) < 1e-3
    q = json.load(open(qos))
    assert q["run_id"] == "TEST1" and q["workload_profile"] == "reduced_load"
    meta = json.load(open(tmp_path / "TEST1_meta.json"))
    assert meta["n_images"] == 500 and meta["model_sha256"]


def test_eval_is_deterministic(tmp_path, tmp_profiles):
    out = tmp_path / "acc.json"
    run("eval", *common(tmp_profiles), "--per-class", "5", "--out", str(out))
    acc = json.load(open(out))
    for v in acc["profiles"].values():
        assert v["deterministic"]
        assert 0 <= v["top1"] <= 1


def test_duplicate_run_id_refused(tmp_path, tmp_profiles):
    run("run", *common(tmp_profiles), "--duration", "2", "--run-id", "DUP", "--out-dir", str(tmp_path))
    p = run("run", *common(tmp_profiles), "--duration", "2", "--run-id", "DUP", "--out-dir", str(tmp_path),
            check=False)
    assert p.returncode != 0 and "unique" in (p.stdout + p.stderr)
