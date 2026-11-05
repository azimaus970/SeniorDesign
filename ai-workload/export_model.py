#!/usr/bin/env python3
"""
export_model.py - export the edge-AI workload model to ONNX and verify it.

Run ONCE on a laptop (needs TensorFlow). The Pi only needs the .onnx file and
model_metadata.json produced here.

The exported model
  * takes float32 NHWC input, RGB pixel values in [0, 255],
  * has dynamic height and width (the same file runs at 224 and 160 px),
  * has its preprocessing built in, and outputs 1000 ImageNet probabilities.

After export the ONNX file is checked against the original Keras model on real
images at every resolution used by the profiles. The export is rejected if the
largest probability difference exceeds --tol (test A1 prerequisite).

Usage:
    pip install -r requirements-export.txt
    python export_model.py --data imagenette2-320/val --out models
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import sys

import numpy as np
import onnxruntime as ort
import tensorflow as tf
import tf2onnx

from dataset import list_images, load_images  # same preprocessing as the Pi

A = tf.keras.applications

# Only models whose Keras constructor accepts a dynamic input shape with
# ImageNet weights are listed (no weight transplanting, which broke
# EfficientNet-B0 in the preliminary benchmark).
MODELS = {
    "mobilenetv3_large": lambda: A.MobileNetV3Large(input_shape=(None, None, 3), weights="imagenet"),
    "mobilenetv3_small": lambda: A.MobileNetV3Small(input_shape=(None, None, 3), weights="imagenet"),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mobilenetv3_large", choices=list(MODELS))
    ap.add_argument("--data", required=True, help="Imagenette val folder (for verification)")
    ap.add_argument("--out", default="models")
    ap.add_argument("--res", type=int, nargs="+", default=[224, 160])
    ap.add_argument("--n-check", type=int, default=50, help="images used for verification")
    ap.add_argument("--tol", type=float, default=1e-3, help="max allowed |p_keras - p_onnx|")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    keras_model = MODELS[args.model]()
    spec = (tf.TensorSpec((None, None, None, 3), tf.float32, name="image"),)
    path = os.path.join(args.out, f"{args.model}.onnx")
    tf2onnx.convert.from_keras(keras_model, input_signature=spec, opset=17, output_path=path)

    # ---- verification against the original model
    items = list_images(args.data, per_class=5)[: args.n_check]
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    checks = {}
    for res in args.res:
        x = load_images(items, res).astype(np.float32)
        pk = keras_model.predict(x, batch_size=16, verbose=0)
        po = np.concatenate([sess.run(None, {in_name: x[i:i + 1]})[0] for i in range(len(x))])
        checks[str(res)] = dict(max_abs_diff=float(np.abs(pk - po).max()),
                                top1_agreement=float((pk.argmax(1) == po.argmax(1)).mean()))
        print(f"{res}px: max |diff| = {checks[str(res)]['max_abs_diff']:.2e}, "
              f"top-1 agreement = {checks[str(res)]['top1_agreement']:.3f}")

    ok = all(c["max_abs_diff"] <= args.tol and c["top1_agreement"] == 1.0 for c in checks.values())
    meta = dict(
        model=args.model, file=os.path.basename(path), sha256=sha256(path),
        size_mb=round(os.path.getsize(path) / 1e6, 2), params_m=round(keras_model.count_params() / 1e6, 2),
        input="float32 NHWC RGB 0-255, dynamic H/W", output="1000 ImageNet probabilities",
        exported_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        tensorflow=tf.__version__, tf2onnx=tf2onnx.__version__, onnxruntime=ort.__version__, opset=17,
        verification=dict(n_images=len(items), tol=args.tol, passed=ok, per_resolution=checks),
    )
    with open(os.path.join(args.out, f"{args.model}.metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"{'PASSED' if ok else 'FAILED'}: {path} ({meta['size_mb']} MB, sha256 {meta['sha256'][:12]}...)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
