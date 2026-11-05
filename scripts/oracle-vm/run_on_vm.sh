#!/usr/bin/env bash
# One-command setup + run of Azim's edge-AI inference app on an Oracle Cloud
# Ampere A1 VM (ARM64 Ubuntu). This VM is a software and ARM speed check only.
# It is NOT a Raspberry Pi 5 and gives no thermal or PCM data.
# Usage (on the VM):  bash run_on_vm.sh
set -euo pipefail
LABEL="Measured on Oracle Cloud Ampere A1 (4 OCPU, 8 GB), not a Pi"
HERE="$(cd "$(dirname "$0")" && pwd)"
APP="$HERE/ai-workload"; [ -d "$APP" ] || APP="$(cd "$HERE/../.." && pwd)/ai-workload"
OUT="$HERE/a1_results"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/console_log.txt") 2>&1
echo "### $LABEL"; echo "### started $(date -u +%FT%TZ)"

echo "== 1/7 system packages"
sudo apt-get update -y
sudo apt-get install -y curl tar ca-certificates

echo "== 2/7 Python 3.13 + pinned packages (same versions as the x86 run)"
if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.13 "$HERE/env"
# shellcheck disable=SC1091
source "$HERE/env/bin/activate"
uv pip install -r "$APP/requirements-pi.txt" pytest

echo "== 3/7 dataset (Imagenette 320 px; only the val split is used)"
cd "$HERE"
if [ ! -d imagenette2-320/val ]; then
  curl -LO https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
  tar xzf imagenette2-320.tgz
fi
DATA="$HERE/imagenette2-320/val"

echo "== 4/7 machine record"
{
  echo "$LABEL"
  echo "recorded_utc: $(date -u +%FT%TZ)"
  echo "--- uname -a"; uname -a
  echo "--- /etc/os-release"; cat /etc/os-release
  echo "--- lscpu"; lscpu
  echo "--- nproc"; nproc
  echo "--- free -h"; free -h
  echo "--- python"; python --version
  echo "--- packages"; python -c "import numpy, onnxruntime, PIL, psutil; print('numpy', numpy.__version__); print('onnxruntime', onnxruntime.__version__); print('pillow', PIL.__version__); print('psutil', psutil.__version__)"
  echo "--- model sha256"; sha256sum "$APP/models/mobilenetv3_large.onnx"
} > "$OUT/machine.txt"
cat "$OUT/machine.txt"

echo "== 5/7 functional tests"
cd "$APP"
DATA="$DATA" python -m pytest -q tests 2>&1 | tee "$OUT/tests.txt" || echo "TESTS FAILED (see tests.txt)"

echo "== 6/7 offline accuracy + determinism + latency per profile (eval)"
python edge_workload.py eval --data "$DATA" --out "$OUT/a1_profile_accuracy.json"

echo "== 7/7 30-minute high_performance run (same settings as the x86 soak)"
python edge_workload.py run --data "$DATA" --run-id SOAK_A1_HP --profile high_performance \
  --pcm-init none --duration 1800 --notes "$LABEL" --out-dir "$OUT"

cd "$HERE"
tar czf a1_results.tgz a1_results
echo "### finished $(date -u +%FT%TZ). Results: $HERE/a1_results.tgz"
