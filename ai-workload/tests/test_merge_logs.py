"""Causal join of the AI sidecar into the main CSV (merge_logs.py).

The times below are arbitrary test inputs that exercise the boundaries; they are
not project data."""
import csv
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "..", "merge_logs.py")


def _write(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _run(tmp_path, main_rows, ai_rows):
    m, a, o = tmp_path / "main.csv", tmp_path / "ai.csv", tmp_path / "out.csv"
    _write(m, ["run_id", "timestamp"], main_rows)
    _write(a, ["run_id", "t_unix", "workload_profile", "inference_fps", "inference_latency_ms"], ai_rows)
    p = subprocess.run([sys.executable, SCRIPT, str(m), str(a), "-o", str(o)], capture_output=True, text=True)
    return p, (list(csv.DictReader(open(o))) if o.exists() else None)


def test_never_uses_a_window_that_ends_after_the_row(tmp_path):
    # main row at 100.2; windows end at 99.9 (past) and 100.9 (future, same floor second)
    p, out = _run(tmp_path,
                  [["R1", "1970-01-01T00:01:40.200+00:00"]],
                  [["R1", "99.9", "balanced", "1", "1"], ["R1", "100.9", "high_performance", "2", "2"]])
    assert p.returncode == 0, p.stderr
    assert out[0]["workload_profile"] == "balanced"


def test_stale_window_and_other_run_are_not_used(tmp_path):
    p, out = _run(tmp_path,
                  [["R1", "1970-01-01T00:01:40.200+00:00"], ["R2", "1970-01-01T00:01:40.200+00:00"]],
                  [["R1", "98.9", "balanced", "1", "1"], ["R3", "100.0", "balanced", "1", "1"]])
    assert p.returncode == 0, p.stderr
    assert out[0]["workload_profile"] == "" and out[1]["inference_fps"] == ""


def test_duplicate_window_end_is_an_error(tmp_path):
    p, _ = _run(tmp_path, [["R1", "1970-01-01T00:01:40.200+00:00"]],
                [["R1", "99.9", "balanced", "1", "1"], ["R1", "99.9", "balanced", "1", "1"]])
    assert p.returncode != 0 and "two AI-log rows" in p.stderr


def test_window_published_after_the_row_is_not_used(tmp_path):
    # window ends at 100.0 (before the row at 100.2) but is published only at 100.3:
    # the controller could not have read it, so the earlier window is used
    m, a, o = tmp_path / "main.csv", tmp_path / "ai.csv", tmp_path / "out.csv"
    _write(m, ["run_id", "timestamp"], [["R1", "1970-01-01T00:01:40.200+00:00"]])
    _write(a, ["run_id", "t_unix", "published_t_unix", "workload_profile", "inference_fps", "inference_latency_ms"],
           [["R1", "99.0", "99.3", "balanced", "1", "1"], ["R1", "100.0", "100.3", "high_performance", "2", "2"]])
    p = subprocess.run([sys.executable, SCRIPT, str(m), str(a), "-o", str(o)], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert list(csv.DictReader(open(o)))[0]["workload_profile"] == "balanced"
