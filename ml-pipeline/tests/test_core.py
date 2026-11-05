"""Logic tests for splits, metrics, calibration energy and E0. See tests/helpers.py: test values are not data."""
import numpy as np
import pandas as pd
import pytest

from thermalml import metrics, splits
from thermalml.estimators import calibration as cal
from thermalml.estimators.e0 import E0Estimator, Endpoints


def test_manifest_validation(tmp_path):
    p = tmp_path / "m.csv"
    pd.DataFrame({"run_id": ["a", "b", "c"], "split": ["train", "validation", ""]}).to_csv(p, index=False)
    with pytest.raises(splits.ManifestError):
        splits.load_manifest(p)
    pd.DataFrame({"run_id": ["a", "a"], "split": ["train", "validation"]}).to_csv(p, index=False)
    with pytest.raises(splits.ManifestError):
        splits.load_manifest(p)
    pd.DataFrame({"run_id": ["a", "b", "c"], "split": ["train", "validation", "test"]}).to_csv(p, index=False)
    m = splits.load_manifest(p)
    assert [h for _, h in splits.leave_one_out(m)] == ["a", "b"]


def test_test_lock(tmp_path):
    lock = splits.TestLock(tmp_path, "x")
    lock.check_and_set({"n": 1})
    with pytest.raises(RuntimeError):
        lock.check_and_set({"n": 2})
    lock.check_and_set({"n": 2}, override=True)


def test_metrics_and_bootstrap():
    assert metrics.mae(np.array([1.0, -3.0])) == 2.0
    assert metrics.rmse(np.array([3.0, -4.0])) == pytest.approx(np.sqrt(12.5))
    mean, lo, hi = metrics.paired_bootstrap(np.array([1.0, 2.0, 3.0, 4.0]), 500, 0.9, 0)
    assert mean == 2.5 and lo <= mean <= hi


def test_compare_variants_sign():
    rows = []
    for i in range(6):
        rows += [dict(run_id=f"r{i}", model="m", horizon_s=10, window_s=30, variant="temperature_only", mae_c=2.0 + 0.1 * i),
                 dict(run_id=f"r{i}", model="m", horizon_s=10, window_s=30, variant="pcm_aware", mae_c=1.0)]
    c = metrics.compare_variants(pd.DataFrame(rows), "m", 10, 30, 500, 0.95, 0)
    assert c["mean_diff"] > 0 and c["signflip_p"] == pytest.approx(2 / 64) and c["improvement_claimed"]


def test_signflip_wilcoxon_exact():
    # all positive, no zeros: p = 2 / 2^n
    assert metrics.signflip_wilcoxon(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])) == pytest.approx(2 / 64)
    assert metrics.signflip_wilcoxon(np.array([1.0, 2.0, 3.0, 4.0, 5.0])) == pytest.approx(2 / 32)
    assert metrics.signflip_wilcoxon(np.zeros(4)) == 1.0
    # symmetric data give p = 1
    assert metrics.signflip_wilcoxon(np.array([-1.0, 1.0])) == 1.0


def test_compare_variants_needs_exact_test_too():
    # three runs all improving: the bootstrap interval is above zero, but the exact
    # test cannot go below 2/8 = 0.25, so no improvement is claimed
    rows = []
    for rid, a, b in [("r1", 2.0, 1.0), ("r2", 3.0, 1.5), ("r3", 2.5, 1.0)]:
        rows += [dict(run_id=rid, model="m", horizon_s=10, window_s=30, variant="temperature_only", mae_c=a),
                 dict(run_id=rid, model="m", horizon_s=10, window_s=30, variant="pcm_aware", mae_c=b)]
    c = metrics.compare_variants(pd.DataFrame(rows), "m", 10, 30, 500, 0.95, 0)
    assert c["ci_low"] > 0 and c["signflip_p"] == pytest.approx(0.25) and not c["improvement_claimed"]


def _cycle(phase, n, heater_w, tfix, fan=1.0):
    return pd.DataFrame({"cycle_id": "c", "elapsed_s": np.arange(n, dtype=float), "phase": phase,
                         "heater_voltage_v": heater_w, "heater_current_a": 1.0, "fixture_temp_c": tfix,
                         "pcm_t1_c": 0.0, "pcm_t2_c": 0.0, "pcm_t3_c": 0.0, "ambient_temp_c": 0.0,
                         "fan_command_pct": fan})


def test_cycle_energy_and_reference_z():
    rig = pd.DataFrame({"fan_command_pct": [1.0, 1.0], "delta_t_fix_amb_k": [0.0, 10.0], "p_loss_w": [0.0, 0.0]})
    c = _cycle("melt", 11, 2.0, 5.0)
    e = cal.cycle_energy(c, rig, c_fix=3.0)
    assert e[-1] == pytest.approx(20.0)              # 2 W for 10 s, no loss, fixture unchanged
    z = cal.reference_z(c, rig, 3.0, euse=20.0)
    assert z[0] == pytest.approx(1.0) and z[-1] == pytest.approx(0.0)
    c2 = c.copy()
    c2["fixture_temp_c"] = np.linspace(5.0, 6.0, 11)  # fixture stores C_fix * 1 K
    assert cal.cycle_energy(c2, rig, 3.0)[-1] == pytest.approx(17.0)


def test_rig_loss_refuses_extrapolation():
    rig = pd.DataFrame({"fan_command_pct": [1.0, 1.0], "delta_t_fix_amb_k": [0.0, 1.0], "p_loss_w": [0.0, 1.0]})
    with pytest.raises(ValueError):
        cal.rig_loss_w(rig, np.array([1.0]), np.array([2.0]))
    assert cal.rig_loss_w(rig, np.array([1.0]), np.array([0.5]))[0] == pytest.approx(0.5)


def _e0():
    return E0Estimator(euse_j=100.0, r_proc_k_per_w=1.0, r_loss_by_fan={1.0: 1.0},
                       endpoints=Endpoints(regenerated_below_c=0.0, regenerated_hold_s=2.0,
                                           depleted_above_c=50.0, depleted_hold_s=2.0))


def test_e0_energy_balance_and_reanchor():
    est = _e0()
    est.start_run("regenerated")
    z = est.step(0, 1.0, soc=11.0, t1=1.0, t2=1.0, t3=1.0, t4=1.0, fan=1.0)   # 10 W in, 0 W out, 1 s
    assert z == pytest.approx(0.9)
    for i in range(3):                                                          # all probes above depleted
        z = est.step(1 + i, 1.0, soc=60.0, t1=60.0, t2=60.0, t3=60.0, t4=60.0, fan=1.0)
    assert z == 0.0 and est.reanchors[-1]["endpoint"] == "depleted"
    assert est.reanchors[-1]["jump"] == pytest.approx(-0.9)


def test_e0_missing_input_stays_invalid_until_endpoint():
    est = _e0()
    est.start_run("regenerated")
    assert np.isnan(est.step(0, 1.0, soc=np.nan, t1=1.0, t2=1.0, t3=1.0, t4=1.0, fan=1.0))
    assert np.isnan(est.step(1, 1.0, soc=2.0, t1=1.0, t2=1.0, t3=1.0, t4=1.0, fan=1.0))
    for i in range(3):
        z = est.step(2 + i, 1.0, soc=-1.0, t1=-1.0, t2=-1.0, t3=-1.0, t4=-1.0, fan=1.0)
    assert z == 1.0 and est.reanchors[-1]["z_before"] is None


def test_e0_unknown_fan_level_is_invalid():
    est = _e0()
    est.start_run("regenerated")
    assert np.isnan(est.step(0, 1.0, soc=2.0, t1=1.0, t2=1.0, t3=1.0, t4=1.0, fan=7.0))
