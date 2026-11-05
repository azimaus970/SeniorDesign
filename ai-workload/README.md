# Edge-AI workload (Azim)

The edge-AI workload for the PCM thermal testbed. It runs MobileNetV3-Large on ONNX Runtime over a fixed, labelled set of 500 Imagenette images, and logs FPS, latency, accuracy, CPU and memory once per second. It follows Part IV Section 5.2.4.1 and Part III Section 4.3.3.2.

All numbers in `results/` were measured on an x86 Linux machine, **not on a Raspberry Pi 5**. Pi 5 figures are still to be measured (tests A1 and A3).

## Files

| File | What it does |
| --- | --- |
| `export_model.py` | Laptop only. Exports MobileNetV3-Large from Keras to ONNX, then checks the ONNX file against the Keras model at 224 and 160 px. It refuses to pass if they disagree. |
| `models/` | The exported `mobilenetv3_large.onnx` and its metadata: hash, versions and verification result. |
| `dataset.py` | The fixed image subset (first 50 files per class, fixed order) and the preprocessing, shared by the laptop and the Pi. |
| `profiles.json` | The three Part III profiles. Only resolution and inference rate change. |
| `edge_workload.py` | The workload: `run`, `eval`, `calibrate` and `set-profile`. |
| `merge_logs.py` | Offline. Fills `workload_profile`, `inference_fps` and `inference_latency_ms` in the Table 4.5 CSV from the AI log. |
| `tests/` | Functional tests on the real model and images. |

## Setup on the Pi 5

```bash
python3 -m venv ~/ai-env && source ~/ai-env/bin/activate
pip install -r requirements-pi.txt
# dataset: Imagenette 320 px, validation split only is used
curl -LO https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz && tar xzf imagenette2-320.tgz
```

Then copy `models/` from this folder to the Pi.

## Before the first real run (on the Pi)

1. **Offline accuracy and determinism per profile (A1):** `python edge_workload.py eval --data imagenette2-320/val --out results/pi5_profile_accuracy.json`
2. **Set the reduced_load rate:** the team agrees the fraction F of balanced FPS. Then run `python edge_workload.py calibrate --data imagenette2-320/val --fraction F --write`. Until this is done, `reduced_load` has no rate and the program refuses to use it. There is deliberately no default value.

## Running

```bash
python edge_workload.py run --data imagenette2-320/val \
    --run-id R001 --pcm-init solid --duration 1800 \
    --control /tmp/ai_control.json --qos-file /tmp/ai_qos.json --out-dir logs
```

- **set_profile():** `python edge_workload.py set-profile balanced --control /tmp/ai_control.json`, or `write_control(path, name)` from Python. The change takes effect at the next 1 s window boundary, after untimed warm-up inferences.
- **get_qos():** `read_qos("/tmp/ai_qos.json")` returns the last completed window's profile, FPS and latency.
- **Fixed schedule:** `--schedule plan.json` with `[{"t": 0, "profile": "high_performance"}, {"t": 600, "profile": "balanced"}]`.
- **Stopping:** Ctrl+C or SIGTERM stops cleanly and writes the summary.

## AI log (`<run_id>_ai.csv`)

There is one row per wall-clock second. Each window ends just after a whole second of the Pi clock (`time.time()`, UTC).

| Column | Meaning |
| --- | --- |
| `run_id`, `utc`, `t_unix`, `t_rel_s`, `win_s` | Run, window end time, and window length |
| `workload_profile` | Profile actually running in this window |
| `profile_requested` | Last profile requested by the schedule or control file |
| `profile_changed_t_unix` | When the running profile last changed |
| `transition` | 1 in the window that contains a change and its warm-up |
| `model`, `res`, `rate_hz`, `threads` | Settings; empty `rate_hz` means back-to-back, with no cap |
| `n_inf`, `inference_fps` | Inferences completed and rate in this window |
| `inference_latency_ms` | Mean latency (maps to Table 4.5) |
| `lat_p50_ms`, `lat_p95_ms`, `lat_max_ms` | Latency distribution |
| `n_correct` | Integer count of correct top-1 predictions among the `n_inf` completed in this window; delivered accuracy over a run is the sum of `n_correct` divided by the sum of `n_inf` |
| `acc_win` | Top-1 accuracy in this window, for checking only; the official accuracy is the offline `eval` result |
| `cpu_proc_pct`, `cpu_sys_pct` | Process CPU (100 = one core) and system CPU |
| `rss_mb`, `mem_used_mb`, `mem_avail_mb` | Process and system memory |
| `pcm_init` | Initial PCM condition given at start |
| `request_t_unix` | Time of the request in force: `requested_t_unix` from the control file, or the schedule time |
| `published_t_unix` | Time taken after this window's QoS was published (QoS file replaced; or just before the CSV row is written when no QoS file is used) |

Missing values are empty, never 0. `merge_logs.py` matches each main-CSV row to the AI row of the same run that was **published** most recently at or before the row time, and less than one control period (1 s) before it, so no AI value that the controller could not yet have read is used. Logs without `published_t_unix` fall back to `t_unix` with a warning. Duplicate publication times are an error, and rows are never matched across runs.

The control file holds the latest desired state, not a queue: a request overwritten before the next 1 s read is never applied, so the controller should log every request it writes. Rejected requests go to the events log. Profile changes take effect only at a window boundary, so no window mixes two profiles; the first window after a change carries the warm-up and has `transition = 1`. The controller should also log the `t_unix` of each QoS window it consumes, so the offline join can be compared with the live decision input.

## Tests

```bash
DATA=imagenette2-320/val pytest -q tests/
```

The tests use a calibration fraction of 0.5 only to exercise the reduced_load code. It is not a project value.
