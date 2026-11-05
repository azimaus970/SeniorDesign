# Phase 3 ML pipeline: thermal predictors and PCM capacity estimators

This folder holds the offline pipeline that trains and evaluates the models of Part III Section 4.3.3.1:

- the short-horizon processor-temperature predictors (persistence and linear-trend baselines, XGBoost, GRU, LSTM)
- the PCM remaining-capacity estimators E0, E1 and E2

It does not contain any data, trained models or results. It only works on the team's real run logs and calibration logs. Every setting is either taken from Part III or Azim's Phase 3 brief (the source is noted beside it in the config), or left empty for the team to fix. No step runs until the values it needs are filled in.

Phase 4 is not built here. That covers controller integration and the M1–M4 / A–D comparison.

## Rules this code follows

| Rule | Where it comes from | How the code applies it |
| --- | --- | --- |
| Whole runs only in train / validation / test | Part III 4.3.3.1 B–C, NFR6 | The team assigns splits in `datasets/split_manifest.csv`; the code only checks the file and never splits rows |
| Test runs used once, after selection | Part III 4.3.3.1 C | Scoring the test split writes a lock file; a second scoring is refused unless explicitly overridden, and the override is recorded |
| Target = `soc_temp_c` shifted by H inside each run | Part III 4.3.3.1 B | The target is written as the change `soc_temp_c(t+H) − soc_temp_c(t)`. `predicted_temp_c` is the current value plus that change |
| Drop targets that span a command, schedule or safety change | Part III 4.3.3.2 | Rows are dropped on any change in fan, cap or profile, on `safety_override`, on unknown commands, and on listed events |
| Hold the last valid value for at most 3 periods, flagged; otherwise invalid | Part III 4.3.3.1 C | The same rule is used when building every training window |
| Scalers fitted on training runs only | Part III 4.3.3.1 B | Fitted on training windows only, then baked into the ONNX graph |
| Raw T1–T3 never enter the predictor; only `pcm_capacity_est` in the PCM-aware variant | Part III 4.3.3.1 B, 4.3.3.2 | The predictor feature list is fixed in config. The M4 variant adds only the estimator output, applied offline |
| Learned estimators exclude `soc_temp_c`, `input_power_w` and the E0 output | Part III 4.3.3.1 A | The estimator feature builder has no access to them |
| Estimator errors in points of z, per test cycle; keep E2 only if it beats E1 by more than the per-cycle spread | Part III 4.3.3.1 A, C | `estimators_cli test` writes the per-cycle table and the rule's outcome |
| Model packages with feature list, scaler, W, H, variant, version, run IDs, manifest hash, library versions, metrics | Part III 4.3.1.2, 4.3.3.1 C | Each package is a folder plus `metadata.json`, indexed by `models/model_metadata.json` |
| Prediction errors are reported per run, then summarized | Part III 4.3.3.1 C | Results go in `results/evaluation_metrics.csv` (one row per run) and the summary files |
| M3 and M4 use the same model type, primary horizon and window | Part III 4.3.3.2 | `predictors train` picks one window per model for both variants: the one whose best settings give the lowest validation RMSE averaged over the variants; each variant then takes its best setting at that window |
| Compared errors use the same target instances | Fair comparison across windows and M3/M4 variants | Validation and test scoring keep only the origins eligible for every compared window and variant (`predictors.common_origins`); each setting's own eligible count stays in `coverage.csv` |

## Layout

```
config/pipeline_config.yaml   all settings, with sources; nulls = team decisions still open
thermalml/
  io.py                Table 4.5 loader, run configs, event logs, throttle-bit split
  splits.py            split manifest checks, manifest hash, leave-one-run-out folds, test lock
  features.py          grid, hold rule, targets, eligibility, flat features, W x F windows
  models/baselines.py  persistence, linear trend
  models/xgb.py        XGBoost predictor
  models/rnn.py        GRU / LSTM predictor, training with early stopping, ONNX export
  metrics.py           MAE, RMSE, per-run tables, paired bootstrap and exact sign-flip Wilcoxon test for the M3/M4 variants
  predictors.py        CLI: train, loro, test
  estimators/calibration.py   calibration package, rig loss, Euse and reference z (Part III 4.2.5 stage D)
  estimators/e0.py            E0 energy balance with endpoint re-anchoring and drift log
  estimators/features.py      estimator features
  estimators/learned.py       E1 (XGBoost or ridge), E2 (stateful GRU/LSTM, ONNX with explicit state)
  estimators_cli.py    CLI: reference, train, test, e0-drift, apply
  package.py           model packages and models/model_metadata.json
  bench.py             on-Pi call timing on a real logged run
  check_config.py      lists the config values still to be fixed
tests/                 logic tests and one end-to-end test on throwaway files
```

## Inputs it needs (none exist yet)

| Input | Path | Format | Who provides it |
| --- | --- | --- | --- |
| Run logs | `data/runs/<run_id>.csv` | Part III Table 4.5. AI fields are filled by `ai-workload/merge_logs.py` | Mazen's logger plus the AI sidecar |
| Run configurations | `data/run_configs/<run_id>.json` | Keys: run_id, control_mode, chamber_setpoint_c, sampling_period_s, initial_pcm_state, cartridge_id, workload_config, schedule_id (Part III 4.3.2) | Whoever runs the experiment |
| Event logs (optional) | `data/events/<run_id>.csv` | Columns `event` and either `elapsed_s` or `t_unix` | RunLogger.log_event |
| Split manifest | `datasets/split_manifest.csv` | run_id, split (train / validation / test), plus mode, setpoint, schedule, initial state, cartridge | Team decision |
| Heater-cycle logs | `calibration/calibration_NNN.csv` | Proposed columns: see `estimators/calibration.py` | Stage D calibration |
| Calibration package | `calibration/pcm_capacity_reference.csv`, `calibration/rig_loss.csv` | Proposed columns: see `estimators/calibration.py` | Stage D calibration |
| Cycle split manifest | `datasets/cycle_split_manifest.csv` | cycle_id, split | Team decision |

The calibration file columns are a proposal. They must be agreed with whoever runs stage D before calibration starts.

## Values the team still has to set

Run `python -m thermalml.check_config config/pipeline_config.yaml` to see the current list. These values were open when the code was written:

- **Missing data:** the missing-data limit (Part III says "a set limit").
- **Events:** the event names that break a target.
- **Predictor features:** the slope windows for predictor features (Part III: "short and long windows").
- **Baselines and selection:** the trend-baseline window and the primary horizon (Part III: fixed after pilot runs).
- **Hyperparameter grids:**
  - the XGBoost grid and early-stopping rounds;
  - the GRU/LSTM grid, which must stay within Part III's 16–64 units and 1–2 layers;
  - the GRU/LSTM learning rate, batch size, epoch limit, patience and seed.
- **Variant comparison:** the bootstrap settings for the paired M3/M4 variant comparison.
- **PCM endpoints and transition range:** fixed after the PCM is selected (Part III).
- **Estimator settings:** the estimator slope windows, and the E1 and E2 grids and training settings.
- **On-Pi timing:** which statistic the limits apply to (Part III reports both median and 95th percentile without naming one) and the per-call pass mark. The combined estimator + predictor limit of 100 ms is Azim's decision.

## Order of use

On the laptop, after real data exist:

```
pip install -r requirements-laptop.txt
python -m thermalml.check_config config/pipeline_config.yaml

# Estimators (after stage D calibration)
python -m thermalml.estimators_cli reference --cartridge <ID>   # reference z per heater cycle; Euse per melt cycle
python -m thermalml.estimators_cli train     --cartridge <ID>   # E1 and E2 on train cycles, chosen on validation cycles
python -m thermalml.estimators_cli test                          # once
python -m thermalml.estimators_cli e0-drift                      # E0 re-anchoring jumps on operational runs
python -m thermalml.estimators_cli apply --estimator <e0 | estimator_E1 | estimator_E2> [--reset-at-endpoints]

# Predictors (after the pilot and scheduled-command runs)
python -m thermalml.predictors train          # all models x horizons x windows x grid; validation metrics; packages
python -m thermalml.predictors loro           # optional stability check of the selected settings
python -m thermalml.predictors test           # once, after selection; writes variant_comparison.csv
```

On the Raspberry Pi 5, with the packages copied over:

```
pip install -r requirements-pi.txt
python -m thermalml.bench --run data/runs/<run_id>.csv --packages <package> [<package> ...] --calls <N>
```

`predictors train` picks, for each model and variant, the setting with the lowest mean per-run validation RMSE at the primary horizon (Part III). The choice of one model type for both M3 and M4 is left to the team. It depends on the validation results of both variants and on the Pi timing.

## Outputs

| File | Content |
| --- | --- |
| `results/evaluation_metrics.csv` | One row per run x model x variant x H x W x setting, for validation and test |
| `results/validation_summary.csv`, `results/test_summary.csv` | Mean and spread across runs |
| `results/coverage.csv` | Eligible rows per run, after each filtering step |
| `results/selection.csv` | Selected setting per model and variant |
| `results/variant_comparison.csv` | Per-run temperature-only minus PCM-aware error, with the bootstrap interval and the exact sign-flip p; an improvement needs both |
| `results/loro_stability.csv` | Leave-one-run-out errors of the selected settings |
| `results/estimator_validation.csv`, `results/estimator_test.csv`, `results/estimator_E2_vs_E1.json` | Estimator errors per cycle and the E2 rule |
| `results/e0_drift.csv` | Each E0 re-anchoring and its jump |
| `results/z_offline/<run_id>.csv` | Estimator output used as the PCM-aware predictor input |
| `models/<package>/`, `models/model_metadata.json` | Packages for the Pi |

## Tests

`python -m pytest -q tests` runs logic tests and one end-to-end test. The values inside the tests only check that the code computes what it should. For example, the slope of a straight ramp must come back exactly. They are not data, settings or results, and the end-to-end test writes only to a temporary folder that is deleted afterwards.

Tested on Python 3.13.18 with numpy 2.5.3, pandas 3.0.5, PyYAML 6.0.3, scikit-learn 1.9.1, joblib 1.6.0, xgboost 3.4.1, torch 2.15.0, onnx 1.23.2, onnxruntime 1.29.0 and pytest 9.2.0. These are the cloud test machine, not the Pi. The Pi install still has to be checked.
