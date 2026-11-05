#!/usr/bin/env python3
"""Reproducible candidate-model benchmark for Part IV Table 5.3.

For each candidate: build the Keras ImageNet model with its own preprocessing
inside the graph (input = float32 RGB 0-255), export to ONNX, check the ONNX
file against Keras on the first 50 subset images, then (in a fresh process that
loads only ONNX Runtime, NumPy and the cached images) count top-1/top-5
correct over the fixed 500-image Imagenette subset and time back-to-back
batch-1 inference with 4 intra-op threads.

Usage: python benchmark.py --data imagenette2-320/val --out results
"""
import argparse, hashlib, json, os, platform, subprocess, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dataset import list_images, load_images, subset_id  # noqa: E402

RES = 224
N_CHECK = 50


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def build(name):
    import tensorflow as tf
    A = tf.keras.applications
    inp = tf.keras.Input((RES, RES, 3), dtype="float32", name="image")
    if name == "mobilenetv3_large":
        m = A.MobileNetV3Large(input_shape=(RES, RES, 3), weights="imagenet")  # preprocessing built in
        out = m(inp)
    elif name == "mobilenetv3_small":
        m = A.MobileNetV3Small(input_shape=(RES, RES, 3), weights="imagenet")
        out = m(inp)
    elif name == "mobilenetv2":
        m = A.MobileNetV2(input_shape=(RES, RES, 3), weights="imagenet")
        out = m(A.mobilenet_v2.preprocess_input(inp))
    elif name == "efficientnet_b0":
        m = A.EfficientNetB0(input_shape=(RES, RES, 3), weights="imagenet")  # rescaling built in
        out = m(inp)
    elif name == "resnet50":
        m = A.ResNet50(input_shape=(RES, RES, 3), weights="imagenet")
        out = m(A.resnet50.preprocess_input(inp))
    else:
        raise ValueError(name)
    full = tf.keras.Model(inp, out)
    return full, int(m.count_params())


def export(name, imgs, out_dir):
    import tensorflow as tf
    import tf2onnx
    import onnxruntime as ort
    model, params = build(name)
    path = os.path.join(out_dir, f"{name}.onnx")
    spec = (tf.TensorSpec((None, RES, RES, 3), tf.float32, name="image"),)
    tf2onnx.convert.from_keras(model, input_signature=spec, opset=17, output_path=path)
    x = imgs[:N_CHECK].astype(np.float32)
    ref = model.predict(x, batch_size=10, verbose=0)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    got = np.concatenate([sess.run(None, {sess.get_inputs()[0].name: x[i:i + 1]})[0] for i in range(len(x))])
    return {"params": params, "onnx_sha256": sha256(path),
            "check_images": N_CHECK,
            "check_max_abs_prob_diff": float(np.max(np.abs(ref - got))),
            "check_top1_agree": int((ref.argmax(1) == got.argmax(1)).sum())}


def vm_hwm_mb():
    """Peak resident set of this process (VmHWM; reset by exec, unlike ru_maxrss)."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024


def run_child(path, data, per_class):
    """Fresh process: only ORT, NumPy, PIL and cached images."""
    import onnxruntime as ort
    items = list_images(data, per_class)
    labels = np.array([l for _, l in items])
    imgs = load_images(items, RES)  # uint8 cache, converted per call
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    for i in range(10):  # untimed warm-up
        sess.run(None, {name: imgs[i:i + 1].astype(np.float32)})
    lat, probs = [], []
    t0 = time.perf_counter()
    for i in range(len(imgs)):
        a = time.perf_counter()
        p = sess.run(None, {name: imgs[i:i + 1].astype(np.float32)})[0][0]
        lat.append((time.perf_counter() - a) * 1000)
        probs.append(p)
    total = time.perf_counter() - t0
    probs = np.array(probs)
    top5 = np.argsort(-probs, axis=1)[:, :5]
    lat = np.array(lat)
    print(json.dumps({
        "n_images": int(len(imgs)),
        "top1_correct": int((top5[:, 0] == labels).sum()),
        "top5_correct": int((top5 == labels[:, None]).any(1).sum()),
        "timed_inferences": int(len(lat)),
        "timed_seconds": total,
        "fps": len(lat) / total,
        "latency_p50_ms": float(np.percentile(lat, 50)),
        "latency_p95_ms": float(np.percentile(lat, 95)),
        "peak_rss_mb": vm_hwm_mb(),
    }))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="+", default=["mobilenetv3_large", "resnet50", "mobilenetv2", "mobilenetv3_small"])
    ap.add_argument("--child", help=argparse.SUPPRESS)
    ap.add_argument("--per-class", type=int, default=50)
    a = ap.parse_args()
    if a.child:
        return run_child(a.child, a.data, a.per_class)
    os.makedirs(a.out, exist_ok=True)
    items = list_images(a.data, a.per_class)
    imgs = load_images(items, RES)
    import onnxruntime, tensorflow, tf2onnx
    cpu = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
    meta = {"machine": {"cpu_model": next((l.split(":", 1)[1].strip() for l in cpu.splitlines() if l.startswith("Model name")), None),
                        "logical_cpus": os.cpu_count(), "platform": platform.platform(), "python": platform.python_version()},
            "versions": {"tensorflow": tensorflow.__version__, "tf2onnx": tf2onnx.__version__,
                         "onnxruntime": onnxruntime.__version__, "numpy": np.__version__},
            "subset": {"n_images": len(items), "per_class": a.per_class, "subset_id": subset_id(items), "resolution": RES},
            "timing": {"threads": 4, "batch": 1, "warmup_untimed": 10, "timed_inferences_per_model": len(items)},
            "script_sha256": sha256(os.path.abspath(__file__)), "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "models": {}}
    for name in a.models:
        r = export(name, imgs, a.out)
        child = subprocess.run([sys.executable, os.path.abspath(__file__), "--data", a.data, "--out", a.out,
                                "--per-class", str(a.per_class), "--child", os.path.join(a.out, f"{name}.onnx")],
                               capture_output=True, text=True, check=True)
        r.update(json.loads(child.stdout.strip().splitlines()[-1]))
        r["top1_pct"] = 100 * r["top1_correct"] / r["n_images"]
        r["top5_pct"] = 100 * r["top5_correct"] / r["n_images"]
        meta["models"][name] = r
        print(name, json.dumps(r), flush=True)
    with open(os.path.join(a.out, "benchmark_results.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
