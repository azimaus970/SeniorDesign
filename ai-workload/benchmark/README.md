# Candidate-model benchmark (Part IV Table 5.3)

Measured on a Linux cloud machine (Intel Xeon @ 2.30 GHz, 4 logical CPUs), not a Raspberry Pi.

- `benchmark.py` builds each Keras ImageNet model with its preprocessing inside the graph, exports it to ONNX,
  checks the ONNX file against Keras on the first 50 subset images, then (in a fresh process holding only ONNX
  Runtime, NumPy and the cached images) counts top-1 and top-5 correct over the fixed 500-image Imagenette
  subset (`dataset.py`, 50 per class) and times batch-1 inference with 4 threads after 10 untimed warm-ups.
- `benchmark_results.json` is the run used in Table 5.3 (all five models). It records the machine, software
  versions, subset id, ONNX hashes, correct counts, FPS, latency p50/p95 and peak resident memory (VmHWM).
- `benchmark_results_first_pass.json` is an earlier full pass of the same script on the same machine without
  EfficientNet-B0. Correct counts are identical; speeds differ, which shows the run-to-run variation.

Re-run:

    pip install -r ../requirements-export.txt psutil
    python benchmark.py --data imagenette2-320/val --out results \
        --models mobilenetv3_large resnet50 mobilenetv2 mobilenetv3_small efficientnet_b0

`benchmark.py` imports `dataset.py` from its own folder: copy `../dataset.py` next to it first.
