"""Logic tests for loading, targets, eligibility and features. See tests/helpers.py: test values are not data."""
import numpy as np
import pandas as pd
import pytest

from helpers import frame, write_run
from helpers import make_test_config as make_cfg
from thermalml.features import build_samples, hold, ls_slope, regularize
from thermalml.io import SchemaError, load_run, throttle_bits
from thermalml.models import baselines


def test_load_keeps_missing_as_nan(tmp_path):
    df = frame(5)
    df.loc[2, "input_power_w"] = np.nan
    run = load_run(write_run(df, tmp_path))
    assert np.isnan(run.loc[2, "input_power_w"])
    assert run["sensor_valid"].all()


def test_load_rejects_missing_column(tmp_path):
    p = write_run(frame(3).drop(columns=["cpu_freq_mhz"]), tmp_path)
    with pytest.raises(SchemaError):
        load_run(p)


def test_regularize_inserts_missing_cycles():
    df = frame(6).drop(index=[3]).reset_index(drop=True)
    g = regularize(df, 1.0)
    assert len(g) == 6 and not g.loc[3, "row_present"]


def test_hold_limit():
    s = pd.DataFrame({"a": [1.0, np.nan, np.nan, np.nan, np.nan, 2.0]})
    held, flag = hold(s, ["a"], 3)
    assert held["a"].tolist()[:4] == [1.0, 1.0, 1.0, 1.0] and np.isnan(held.loc[4, "a"])
    assert flag["a"].tolist() == [False, True, True, True, False, False]


def test_throttle_bits():
    b = throttle_bits(pd.Series([0b1010, np.nan]), [0, 1, 3])
    assert b.iloc[0].tolist() == [0, 1, 1] and b.iloc[1].isna().all()


def test_target_is_delta_and_ramp_features_exact():
    cfg = make_cfg()
    run = load_run_frame(frame(60))
    s = build_samples(run, cfg, 10, 30, trend_window_s=5)
    np.testing.assert_allclose(s.y_delta, 1.0)                    # 0.1 per s over 10 s
    np.testing.assert_allclose(s.flat["soc_temp_c_slope3s"], 0.1, atol=1e-12)
    np.testing.assert_allclose(baselines.linear_trend(s, 1.0), 1.0, atol=1e-12)
    np.testing.assert_allclose(baselines.persistence(s), 0.0)
    np.testing.assert_allclose(s.flat["soc_temp_c"] - s.flat["soc_temp_c_lag5s"], 0.5, atol=1e-12)
    assert s.elapsed_s.min() == 29 and s.elapsed_s.max() == 49   # full window, and t+H inside run


def test_command_change_drops_intervals():
    cfg = make_cfg()
    df = frame(60)
    df.loc[40:, "fan_command_pct"] = 2.0          # change at t = 40
    s = build_samples(load_run_frame(df), cfg, 10, 30)
    t = s.elapsed_s
    assert not np.any((t >= 30) & (t <= 39))       # intervals [t, t+10] containing the change
    assert 29 in t


def test_safety_override_and_event_drop():
    cfg = make_cfg()
    df = frame(60)
    df.loc[45, "safety_override"] = "true"
    ev = pd.DataFrame({"elapsed_s": [33.0], "event": ["schedule_change"]})
    s = build_samples(load_run_frame(df), cfg, 10, 30, events=ev)
    t = set(s.elapsed_s.tolist())
    assert not (t & set(range(35, 46))) and not (t & set(range(23, 34)))


def test_missing_target_and_window_rules():
    cfg = make_cfg(**{"missing_data.max_missing_fraction": 0.0})
    df = frame(80)
    df.loc[50, "cpu_usage_pct"] = np.nan          # held -> windows containing it are rejected at fraction 0
    df.loc[70, "soc_temp_c"] = np.nan             # target at t = 60 missing
    s = build_samples(load_run_frame(df), cfg, 10, 30)
    t = set(s.elapsed_s.tolist())
    assert not (t & set(range(50, 69 + 1)))       # windows covering t=50 end at 50..79
    assert 49 in t


def test_pcm_aware_needs_offline_z():
    cfg = make_cfg()
    with pytest.raises(ValueError):
        build_samples(load_run_frame(frame(60)), cfg, 10, 30, variant="pcm_aware")


def test_sequence_window_shape():
    cfg = make_cfg()
    s = build_samples(load_run_frame(frame(60)), cfg, 10, 30, with_sequences=True)
    assert s.seq.shape == (len(s.y_delta), 30, len(s.seq_fields))
    assert s.seq[0, -1, s.seq_fields.index("soc_temp_c")] == pytest.approx(s.current_temp[0])


def test_ls_slope():
    assert ls_slope(np.array([[0.0, 2.0, 4.0, 6.0]]), 2.0)[0] == pytest.approx(1.0)


def load_run_frame(df):
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        return load_run(write_run(df, Path(d)))


def test_common_origins_score_every_window_on_the_same_targets():
    from thermalml.predictors import common_origins, restrict
    cfg = make_cfg()
    df = load_run_frame(frame(200))
    runs = {"T1": (df, None)}
    common = common_origins({"temperature_only": runs}, cfg, 10, [30, 60])
    s30 = build_samples(df, cfg, 10, 30)
    s60 = build_samples(df, cfg, 10, 60)
    assert len(s30.y_delta) > len(s60.y_delta)            # the shorter window could start earlier
    r30, r60 = restrict([s30], common)[0], restrict([s60], common)[0]
    assert r30.elapsed_s.tolist() == r60.elapsed_s.tolist() == s60.elapsed_s.tolist()
    assert len(r30.flat) == len(r30.y_delta) and r30.coverage["scored_common_origins"] == len(r30.y_delta)
