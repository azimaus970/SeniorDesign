"""
dataset.py - the fixed, labelled image subset used by the edge-AI workload.

Imagenette (fast.ai) is a 10-class subset of ImageNet. Its validation split is
used; images are chosen deterministically (sorted file names, first N per
class) and interleaved in a fixed order, so every run sees exactly the same
inputs in the same order. No random numbers are used anywhere.

Preprocessing (identical on laptop and Pi): resize the shorter side to
res * 256 / 224 with bilinear filtering, then centre-crop res x res.
"""
import hashlib
import os

import numpy as np
from PIL import Image

# Imagenette folder name -> ImageNet-1k class index
IMAGENETTE_TO_IMAGENET = {
    "n01440764": 0,    # tench
    "n02102040": 217,  # English springer
    "n02979186": 482,  # cassette player
    "n03000684": 491,  # chain saw
    "n03028079": 497,  # church
    "n03394916": 566,  # French horn
    "n03417042": 569,  # garbage truck
    "n03425413": 571,  # gas pump
    "n03445777": 574,  # golf ball
    "n03888257": 701,  # parachute
}


def list_images(root, per_class=50):
    """Return [(path, imagenet_label)] in a fixed order. per_class=0 means all."""
    per = []
    for wnid in sorted(IMAGENETTE_TO_IMAGENET):
        d = os.path.join(root, wnid)
        if not os.path.isdir(d):
            raise FileNotFoundError(f"missing class folder {d}")
        files = sorted(f for f in os.listdir(d) if f.lower().endswith((".jpeg", ".jpg", ".png")))
        if per_class > 0:
            if len(files) < per_class:
                raise ValueError(f"{d} has only {len(files)} images, need {per_class}")
            files = files[:per_class]
        per.append([(os.path.join(d, f), IMAGENETTE_TO_IMAGENET[wnid]) for f in files])
    # round-robin over classes so every second of logging sees a class mix
    items = []
    for i in range(max(len(p) for p in per)):
        items += [p[i] for p in per if i < len(p)]
    return items


def subset_id(items):
    """Short hash of the file list and labels, recorded in run metadata."""
    h = hashlib.sha256()
    for p, lab in items:
        h.update(f"{os.path.basename(os.path.dirname(p))}/{os.path.basename(p)}:{lab}\n".encode())
    return h.hexdigest()[:16]


def load_images(items, res):
    """Decode and preprocess all images to a uint8 array (N, res, res, 3)."""
    resize_to = int(round(res * 256 / 224))
    out = np.empty((len(items), res, res, 3), dtype=np.uint8)
    for i, (p, _) in enumerate(items):
        with Image.open(p) as im:
            im = im.convert("RGB")
            w, h = im.size
            s = resize_to / min(w, h)
            im = im.resize((max(res, round(w * s)), max(res, round(h * s))), Image.BILINEAR)
            w, h = im.size
            left, top = (w - res) // 2, (h - res) // 2
            out[i] = np.asarray(im.crop((left, top, left + res, top + res)))
    return out
