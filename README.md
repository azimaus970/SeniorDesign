# PCM-cooled Raspberry Pi 5: edge-AI and ML work (Azim)

This repository holds Azim's part of the senior design project: the edge-AI workload, PCM capacity estimation, thermal prediction and model evaluation. The other parts of the project (PCM cartridge, chamber and instrumentation, Pi integration and supervisory controller) belong to teammates and are not here.

## Status, stated plainly

- **The team does not have the Raspberry Pi 5 yet.** No result in this repository was measured on a Pi.
- Every measured result says which machine it came from. The results that exist were measured on a 4-core Intel Xeon Linux cloud machine (x86), as recorded in each folder's `test_record.txt` and in the result files' metadata.
- The thermal prediction and PCM estimator pipeline contains **no data, no trained models and no results**. It runs only on the team's real run logs once they exist. Settings that Part III or Azim's brief do not fix are left empty in the config as open team decisions.
- Nothing here uses made-up numbers, synthetic data or estimates.

## Contents

| Folder | What it is |
| --- | --- |
| [`ai-workload/`](ai-workload/) | The edge-AI inference app: MobileNetV3-Large on ONNX Runtime over a fixed 500-image Imagenette subset, three Part III workload profiles, a per-second AI log, a control file for the controller, and a merge tool for the shared Table 4.5 CSV. Includes the exported `.onnx` model, the candidate-model benchmark, functional tests, and a 30-minute soak run (x86 cloud machine, not a Pi). See its [README](ai-workload/README.md). |
| [`ml-pipeline/`](ml-pipeline/) | The offline Phase 3 pipeline (`thermalml`): persistence and linear-trend baselines, XGBoost, GRU and LSTM temperature predictors at 10/20/30 s horizons, PCM capacity estimators E0 (energy balance), E1 and E2, run-level splits, metrics, packaging for the Pi, and tests. See its [README](ml-pipeline/README.md). |
| [`part4/`](part4/) | Figures for Part IV Sections 5.1.4 and 5.2.4 (Mermaid `.mmd` sources with rendered PNG/SVG), plus the review dispositions, audit findings and template check written while preparing that section. |
| [`diagrams/`](diagrams/) | The Pi 5 thermal loop diagram (Mermaid source and renders). |
| [`scripts/oracle-vm/`](scripts/oracle-vm/) | `run_on_vm.sh`: one-command setup and run of the inference app on an Oracle Cloud Ampere A1 (ARM64) VM. That VM is an ARM software and speed check only; it is not a Pi 5 and gives no thermal or PCM data. |

## Not included

The downloaded Imagenette dataset (fetch it as described in `ai-workload/README.md`), Python caches, Word documents, and teammates' work.

## Running the tests

```bash
# Inference app (needs the Imagenette 320 px validation split)
cd ai-workload && pip install -r requirements-pi.txt pytest
DATA=<path to imagenette2-320/val> python -m pytest -q tests

# ML pipeline
cd ml-pipeline && pip install -r requirements-laptop.txt pytest
python -m pytest -q tests
python -m thermalml.check_config   # lists the settings still open for the team
```
