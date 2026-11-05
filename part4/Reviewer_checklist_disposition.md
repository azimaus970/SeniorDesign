# Reviewer checklist (R01–R48): what was accepted, what was not

I checked each point against Part III, the actual code and the measured results. **Accepted** means the point was correct and is now in the doc (or the code). **Already met** means the doc or code already did it, or Part III already owns it; at most a clarifying sentence was added. **Rejected** means the point is wrong, or acting on it would break your no-made-up-numbers rule.

## Points that changed results or code

| # | Verdict | Why |
|---|---|---|
| R01 | **Accepted: the reviewer was right** | The old Table 5.3 accuracies (82.1, 81.3, 77.3, 74.9) cannot come from 500 images, because 500 images only allow steps of 0.2 points. No record of that first benchmark exists, so those values were withdrawn. I re-ran all five models on a cloud Linux machine (Intel Xeon @ 2.30 GHz, 4 CPUs; not a Pi). Every value is now shown with its count, e.g. 412/500 = 82.4%. That matches the inference app's 412/500, so the 82.1 vs 82.4 question is settled. The script and results file are in `ai-workload/benchmark/`. |
| R01 (consequence) | **Correction** | The re-run shows **ResNet50 was slightly more accurate (416 vs 412 of 500)**. The old claim that MobileNetV3-Large had the highest accuracy was false and has been removed. MobileNetV3-Large is still selected because ResNet50 has about 2.8× the latency and 1.9× the memory for 4 more images out of 500. |
| R02 | Accepted (resolved by data) | With a clean export, EfficientNet-B0 passes the equivalence check and gets 395/500. It is now ranked normally, and the old "export fault" sentence is gone. |
| R03 | Accepted | Both machines are now named precisely: the benchmark machine and the application machine (Xeon 2.8 GHz VM). Versions, subset id and hashes are in the results file, and statuses say exactly what was tested where. A second benchmark pass varied up to 21% in FPS, and the doc now says so. |
| R04 | **Accepted: real code bug** | `merge_logs.py` matched on the same floor-second. A main row at 100.2 s could take an AI window ending at 100.9 s, which uses data from the future. It is now a causal join: the latest window ending at or before the row, at most 1 s old. Duplicate times are now an error. Three new tests pass. |
| R28 | **Accepted: real code issue** | The pipeline compared windows (30/60/120 s) and the M3/M4 variants on different sets of rows. I added `common_origins()` so every compared error uses the same targets. Coverage is still reported per setting. A new test was added, and all 24 pipeline tests pass. |

## Points accepted into the text

| # | Verdict | Why |
|---|---|---|
| R05 | Accepted, partly | The doc now states the direction in which d = M4 − M3 counts as better for every metric (new column in Table 5.2). It also gives a verdict rule for mixed results and separates statistical evidence from usefulness. **Not done:** choosing the primary outcome and the "useful difference" value. That is a team decision, so the doc says it is recorded before the first comparison run. |
| R06 | Accepted | One primary outcome is confirmatory; every other metric and every per-setpoint result is exploratory. Pairs are pooled across setpoints. |
| R07 | Accepted | The old text said M1 vs M2 shows "the effect of the PCM hardware". That overclaims, because M1 and M2 also differ in the whole cooling assembly and the fan behaviour. It is now described as a whole-system effect. |
| R08 | Accepted | The requested schedule (same for all modes) is now separated from the realized profile, fan and cap, which are logged with their effective times. |
| R09 | Accepted | The six-pair minimum is now described honestly as a significance floor only, saying nothing about power. The final count comes from pilot variability. |
| R10 | Accepted | Statistic (mean of d), bootstrap method (percentile, resampling whole pairs), and Wilcoxon zero/tie handling (exact sign-flip, Pratt) are now defined. |
| R11 | Accepted | Runs are now classed as setup-invalid (may be repeated) or controller outcome (never replaced). Reliability is reported per mode from attempted runs. |
| R12 | Accepted | Time to first throttling is stored with an event indicator. The paired comparison uses throttling-free time within the common observation period, with no invented event times. |
| R13 | Accepted | 224 px and 160 px both run back-to-back on 4 threads, so the heat ordering is now something A1 measures, not an assumption. |
| R14 | Accepted | The ONNX Runtime spinning setting is recorded, since it affects idle power in the capped profile. |
| R15 | Accepted | The regenerated-endpoint start condition can't apply to M1 (no PCM), so it now applies to M2–M4 only. A2 now also checks regeneration at all three setpoints. |
| R16 | Accepted | The fallback only lengthens phases while depletion is still progressing; a stall goes to team review. |
| R17 | Accepted | E0's correction is reported before each reset. |
| R18 | Accepted | Accuracy claims between endpoints are limited to the heater-cycle domain. |
| R19 | Accepted | The "per-cycle spread" rule is now defined exactly as the code computes it (mean improvement > sample SD). Choosing a model is separated from proving it is good enough for threshold decisions. |
| R20 | Accepted, partly | C1 now checks for similar inputs at different reference z. I did not claim the sensors are inadequate; there is no data yet. |
| R21 | Accepted | The fallback order is now frozen in the config before test access, and results are reported whether favourable or not. |
| R22 | Accepted | The claim is now stated as unseen runs on the same rig, with no claim about runs recorded later in time. |
| R24 | Accepted | The forecast is now described as conditional on unchanged commands; the share of excluded rows is reported, and benefit is judged in D1. |
| R29 | Accepted | E2 streaming contract added: one new step per cycle, h and c state for LSTM, a repeat call doesn't advance the state. The code already exports both LSTM states. |
| R32 | Accepted | Every shipped package gets a parity check, and an XGBoost→ONNX fallback is only used after it passes. |
| R33 | Accepted | The app logs per-second statistics only, so "latency p95" is renamed honestly (it is not a whole-run p95). FPS and energy now use inference counts. |
| R34 | Accepted | "Time-weighted effective accuracy" was wrong when FPS differs between profiles. It is replaced by delivered accuracy = correct ÷ completed, from the AI log's counts. |
| R35 | Accepted | C4 now reports the FPS loss together with the power change. |
| R37 | Accepted | Signed error, missed warnings and unnecessary warnings above T_warn are added. |
| R38 | Accepted | The 100 ms limit had been listed as "still to set" in one place; that inconsistency is fixed. C4 now also checks the full cycle and late cycles, since Part III stops a run on an overrun. |
| R39 | Accepted | Lead time is now measured from the start of each high-load phase to the first applied action. |
| R40 | Accepted | AM6 now has an M4 replay test as well as M3. The doc states that replay checks the software, not the temperatures. |
| R41 | Accepted | A3 now has a defined scope: system-wide memory, per-process RSS, swap, every control period, for a full run. |
| R42 | Accepted | Accuracy, FPS and latency limits are fixed from what the application needs before A1, not afterwards. |
| R43 | Accepted | Energy per inference is now reported with terminal z and described as an electrical efficiency over the fixed run. |
| R44 | Accepted | One-period detection resolution is stated. The old text said under-voltage throttling "is not thermal"; that was wrong, and it now says the exclusion only removes a confounder. |
| R45 | **Accepted: real loops** | AM1 needed full-stack A3, which needs models from later milestones. AM2 needed pilot logs, which come from AM3. A3 is moved to AM6 and B1-on-pilot-logs to AM3, and both figures are redrawn. |
| R47 | Accepted, partly | "Following Bastida" wrongly implied they reset state at endpoints; it is now worded as motivation only. I did **not** add the reviewer's claim that Bastida is simulation-based, because I can't verify it. |
| R48 | Accepted | Both figures are redrawn top-to-bottom with larger labels, and table column widths are fixed. |

## Already met (small clarification added at most)

| # | Why |
|---|---|
| R23 | Already true in the code: the estimator is trained only on heater cycles, so no reference z or predictor test data reaches the predictor. One sentence added. |
| R25 | Already true in the code: targets and windows are built on a time grid from elapsed seconds, and a duplicate grid point is an error. One sentence added. |
| R26 | Already true in the code: a target only counts if it was measured, not held, at both t and t+H. One sentence added. |
| R27 | Already true: the same measured temperature is used in the target and in the reconstructed prediction. One sentence added. |
| R30 | Part III Section 4.3.3.2 already owns readiness and recovery (is_ready, model error ends the run). The doc now references it instead of repeating it. |
| R31 | The control file is already replaced atomically, and the AI log already records the profile actually running. The doc now says this and that the file holds the latest state, not a queue. Request ids were not added. |
| R36 | Already true: warm-up time stays in the FPS denominator. Now stated explicitly. |

## Rejected

| # | Why |
|---|---|
| R46 (dates part) | Owners are already listed. Adding milestone dates would be made-up numbers; calendar dates belong in the team timeline (5.2.5). |
| R48 (rewrite the opening) | Not done. The cross-references in the opening are there so that Part III isn't repeated, which you asked for. |
| Reviewer's sample values | Everywhere the reviewer implied a number (tolerances, resample counts, useful effect size), I left the value to be fixed by the team, per your rule. |
