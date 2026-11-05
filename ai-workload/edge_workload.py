#!/usr/bin/env python3
"""
edge_workload.py - edge-AI inference workload for the PCM thermal testbed.

MobileNetV3-Large (ONNX Runtime, CPU, 4 threads) classifies a fixed 500-image
Imagenette subset in an endless loop. Three workload profiles from Part III
(high_performance, balanced, reduced_load) change only the input resolution
and the inference rate.

Sub-commands
  run          run the workload and write the per-second AI sidecar log
  eval         offline accuracy of every profile + determinism check (test A1)
  calibrate    measure balanced FPS on this board and set reduced_load's rate
  set-profile  request a profile change in a running workload (control file)

Interfaces for the controller (Mazen's WorkloadManager)
  set_profile(name)  -> python edge_workload.py set-profile NAME --control FILE
                        or write_control(FILE, NAME) from Python. The running
                        workload re-reads the file at every 1 s window boundary.
  get_qos()          -> read_qos(FILE): JSON rewritten every second with the
                        last window's profile, FPS and latency.

Output files per run (in --out-dir)
  <run_id>_ai.csv       one row per wall-clock second (sidecar to Table 4.5)
  <run_id>_events.csv   start, profile changes, control errors, stop
  <run_id>_meta.json    run configuration, versions, dataset and model hashes
  <run_id>_summary.json totals at the end

Join to the Table 4.5 CSV causally on run_id and published_t_unix (seconds, same
Pi clock): t_unix is the end of the window, published_t_unix the time after the
row's QoS was published (QoS file replaced, or CSV row about to be written when
no QoS file is used), so a window counts only from when the controller could read
it. request_t_unix is the time of the request in force (control-file
requested_t_unix or schedule time), tying each applied profile to its request.
Column mapping: workload_profile -> workload_profile, inference_fps ->
inference_fps, inference_latency_ms -> inference_latency_ms. Accuracy stays
in the sidecar only. Missing values are written empty, never as 0.
"""
import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import resource
import signal
import statistics
import sys
import time

import numpy as np
import onnxruntime as ort
import psutil

from dataset import list_images, load_images, subset_id

HERE = os.path.dirname(os.path.abspath(__file__))
SIDECAR_FIELDS = [
    "run_id", "utc", "t_unix", "t_rel_s", "win_s",
    "workload_profile", "profile_requested", "profile_changed_t_unix", "transition",
    "model", "res", "rate_hz", "threads",
    "n_inf", "inference_fps", "inference_latency_ms", "lat_p50_ms", "lat_p95_ms", "lat_max_ms",
    "n_correct", "acc_win", "cpu_proc_pct", "cpu_sys_pct", "rss_mb", "mem_used_mb", "mem_avail_mb", "pcm_init",
    "request_t_unix", "published_t_unix",
]


# ----------------------------------------------------------------------------- helpers
def utc_iso(t):
    return dt.datetime.fromtimestamp(t, dt.timezone.utc).isoformat(timespec="milliseconds")


def atomic_write_json(path, obj):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)  # readers never see a half-written file


def write_control(path, profile):
    """Controller side of set_profile(): request a profile change."""
    atomic_write_json(path, {"profile": profile, "requested_t_unix": round(time.time(), 3)})


def read_qos(path):
    """Controller side of get_qos(): last completed window, or None."""
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def load_profiles(path):
    with open(path) as f:
        cfg = json.load(f)
    for name, p in cfg["profiles"].items():
        if p.get("rate_hz") is not None and p["rate_hz"] <= 0:
            raise ValueError(f"profile {name}: rate_hz must be > 0 or null")
    return cfg


def check_usable(profiles, name):
    if profiles[name].get("needs_rate"):
        sys.exit(f"profile {name} has no measured rate yet; run 'calibrate --fraction F --write' on the Pi first")


def fmt(x, nd=2):
    """Empty string for missing values, never 0."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return round(x, nd) if isinstance(x, float) else x


def make_session(model_path, threads):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])


def model_info(models_dir, model):
    path = os.path.join(models_dir, f"{model}.onnx")
    if not os.path.exists(path):
        sys.exit(f"model not found: {path} (run export_model.py first)")
    meta_path = os.path.join(models_dir, f"{model}.metadata.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    if meta and meta.get("verification", {}).get("passed") is False:
        sys.exit(f"{meta_path}: export verification FAILED, refusing to run")
    return path, meta


class Workload:
    """Model session + preprocessed images for every profile resolution."""

    def __init__(self, args, cfg):
        self.cfg = cfg
        self.threads = cfg["threads"]
        self.model = cfg["model"]
        self.model_path, self.model_meta = model_info(args.models_dir, self.model)
        self.items = list_images(args.data, args.per_class)
        self.labels = np.array([lab for _, lab in self.items])
        self.n = len(self.items)
        self.sess = make_session(self.model_path, self.threads)
        self.in_name = self.sess.get_inputs()[0].name
        # preload every resolution so a profile change never touches the disk
        self.images = {}
        for p in cfg["profiles"].values():
            if p["res"] not in self.images:
                self.images[p["res"]] = load_images(self.items, p["res"])

    def infer(self, res, idx):
        x = self.images[res][idx:idx + 1].astype(np.float32)
        return self.sess.run(None, {self.in_name: x})[0][0]

    def warmup(self, res, k=10):
        for i in range(k):
            self.infer(res, i % self.n)


# ----------------------------------------------------------------------------- run
_stop = False


def _on_signal(signum, frame):
    global _stop
    _stop = True


def cmd_run(args):
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    cfg = load_profiles(args.profiles)
    profiles = cfg["profiles"]
    start_profile = args.profile or cfg["default_profile"]
    if start_profile not in profiles:
        sys.exit(f"unknown profile {start_profile}; choose from {list(profiles)}")
    check_usable(profiles, start_profile)

    run_id = args.run_id or dt.datetime.now().strftime("R%Y%m%d_%H%M%S")
    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.join(args.out_dir, run_id)
    if os.path.exists(f"{base}_ai.csv"):
        sys.exit(f"{base}_ai.csv already exists; run IDs must be unique")
    schedule = sorted(json.load(open(args.schedule)), key=lambda e: e["t"]) if args.schedule else []
    for e in schedule:
        if e["profile"] not in profiles:
            sys.exit(f"schedule: unknown profile {e['profile']}")
        check_usable(profiles, e["profile"])

    print(f"[{run_id}] loading model and images ...", flush=True)
    wl = Workload(args, cfg)
    proc = psutil.Process()

    t_start = time.time()
    meta = dict(
        run_id=run_id, start_utc=utc_iso(t_start), start_t_unix=round(t_start, 3),
        pcm_init=args.pcm_init, notes=args.notes, host=os.uname().nodename,
        python=sys.version.split()[0], onnxruntime=ort.__version__, numpy=np.__version__,
        model=wl.model, model_file=os.path.basename(wl.model_path), model_sha256=wl.model_meta.get("sha256", ""),
        threads=wl.threads, dataset=os.path.abspath(args.data), n_images=wl.n,
        subset_id=subset_id(wl.items), profiles=profiles, start_profile=start_profile,
        schedule=schedule, control_file=args.control, duration_s=args.duration,
    )
    with open(f"{base}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    fcsv = open(f"{base}_ai.csv", "w", newline="")
    w = csv.DictWriter(fcsv, fieldnames=SIDECAR_FIELDS)
    w.writeheader()
    fev = open(f"{base}_events.csv", "w", newline="")
    ev = csv.writer(fev)
    ev.writerow(["run_id", "utc", "t_unix", "event", "detail"])

    def event(name, detail=""):
        t = time.time()
        ev.writerow([run_id, utc_iso(t), round(t, 3), name, detail])
        fev.flush()
        print(f"  {utc_iso(t)} {name} {detail}", flush=True)

    event("start", start_profile)
    active, requested = None, start_profile
    request_t = t_start
    changed_t = None
    ctrl_mtime = None
    sched_i = 0
    idx = 0
    tot = top1 = top5 = 0
    pred_hash = hashlib.sha256()
    psutil.cpu_percent(None)
    proc.cpu_percent(None)

    # windows end on whole wall-clock seconds so rows join to 1 Hz telemetry
    next_t = time.time()
    win_start = time.time()
    win_end = math.floor(win_start) + 1.0
    end_t = t_start + args.duration if args.duration > 0 else float("inf")

    try:
        while not _stop and win_start < end_t:
            # ---- requested profile: schedule first, then the live control file
            t_rel = win_start - t_start
            while sched_i < len(schedule) and schedule[sched_i]["t"] <= t_rel:
                requested = schedule[sched_i]["profile"]
                request_t = t_start + schedule[sched_i]["t"]
                sched_i += 1
            if args.control and os.path.exists(args.control):
                m = os.stat(args.control).st_mtime_ns
                if m != ctrl_mtime:
                    ctrl_mtime = m
                    try:
                        cj = json.load(open(args.control))
                        req = cj["profile"]
                        if req not in profiles:
                            raise KeyError(req)
                        if profiles[req].get("needs_rate"):
                            raise ValueError(f"{req} has no measured rate yet")
                        requested = req
                        request_t = cj.get("requested_t_unix", time.time())
                    except (OSError, ValueError, KeyError, TypeError) as e:
                        event("control_error", f"{type(e).__name__}: {e}")
            transition = 0
            if requested != active:
                event("profile_change", f"{active} -> {requested}")
                active = requested
                changed_t = time.time()
                wl.warmup(profiles[active]["res"])  # untimed; this window is flagged
                transition = 1

            res = profiles[active]["res"]
            rate = profiles[active]["rate_hz"]
            period = 1.0 / rate if rate else 0.0

            # ---- one window
            lats, ok = [], 0
            if transition:
                next_t = time.time()
            while not _stop:
                now = time.time()
                if now >= win_end:
                    break
                if period:
                    if next_t > now:
                        time.sleep(min(next_t - now, win_end - now))
                        continue
                    next_t = max(next_t + period, now)  # if behind, no burst catch-up
                t0 = time.perf_counter()
                p = wl.infer(res, idx)
                lats.append((time.perf_counter() - t0) * 1000.0)
                top = np.argpartition(p, -5)[-5:]
                pred = int(top[np.argmax(p[top])])
                y = wl.labels[idx]
                ok += pred == y
                top1 += pred == y
                top5 += y in top
                tot += 1
                pred_hash.update(pred.to_bytes(2, "little"))
                idx = (idx + 1) % wl.n

            t_end = time.time()
            win_s = t_end - win_start
            mem = psutil.virtual_memory()
            n_inf = len(lats)
            q = statistics.quantiles(lats, n=20, method="inclusive") if n_inf >= 2 else None
            row = dict(
                run_id=run_id, utc=utc_iso(t_end), t_unix=round(t_end, 3), t_rel_s=round(t_end - t_start, 3),
                win_s=round(win_s, 3), workload_profile=active, profile_requested=requested,
                profile_changed_t_unix=fmt(changed_t, 3), transition=transition,
                model=wl.model, res=res, rate_hz=fmt(rate, 2), threads=wl.threads,
                n_inf=n_inf, inference_fps=fmt(n_inf / win_s if win_s > 0 else None),
                inference_latency_ms=fmt(statistics.fmean(lats) if lats else None),
                lat_p50_ms=fmt(statistics.median(lats) if lats else None),
                lat_p95_ms=fmt(q[18] if q else (lats[0] if lats else None)),
                lat_max_ms=fmt(max(lats) if lats else None),
                n_correct=int(ok),
                acc_win=fmt(ok / n_inf if n_inf else None, 4),
                cpu_proc_pct=fmt(proc.cpu_percent(None), 1), cpu_sys_pct=fmt(psutil.cpu_percent(None), 1),
                rss_mb=fmt(proc.memory_info().rss / 2**20, 1),
                mem_used_mb=fmt((mem.total - mem.available) / 2**20, 1), mem_avail_mb=fmt(mem.available / 2**20, 1),
                pcm_init=args.pcm_init, request_t_unix=fmt(request_t, 3),
            )
            if args.qos_file:
                atomic_write_json(args.qos_file, {k: row[k] for k in (
                    "run_id", "t_unix", "workload_profile", "profile_changed_t_unix", "transition",
                    "inference_fps", "inference_latency_ms", "lat_p95_ms", "n_inf", "request_t_unix")})
            row["published_t_unix"] = fmt(time.time(), 3)   # taken after publication: never early
            w.writerow(row)
            fcsv.flush()
            win_start, win_end = t_end, math.floor(t_end) + 1.0
    finally:
        event("stop", "signal" if _stop else "duration reached")
        fcsv.close()
        fev.close()
        summ = dict(
            run_id=run_id, duration_s=round(time.time() - t_start, 1), inferences=tot,
            top1=fmt(top1 / tot if tot else None, 4), top5=fmt(top5 / tot if tot else None, 4),
            prediction_sha256=pred_hash.hexdigest()[:16],
            peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        )
        with open(f"{base}_summary.json", "w") as f:
            json.dump(summ, f, indent=2)
        print(json.dumps(summ), flush=True)


# ----------------------------------------------------------------------------- eval
def cmd_eval(args):
    """Offline accuracy per profile; every image once per pass; repeated passes must match."""
    cfg = load_profiles(args.profiles)
    wl = Workload(args, cfg)
    out = dict(model=wl.model, model_sha256=wl.model_meta.get("sha256", ""), subset_id=subset_id(wl.items),
               n_images=wl.n, threads=wl.threads, host=os.uname().nodename, onnxruntime=ort.__version__,
               evaluated_utc=utc_iso(time.time()), profiles={})
    for name, p in cfg["profiles"].items():
        res = p["res"]
        wl.warmup(res)
        hashes, lats = [], []
        for r in range(args.repeats):
            preds, t5 = [], 0
            for i in range(wl.n):
                t0 = time.perf_counter()
                pr = wl.infer(res, i)
                lats.append((time.perf_counter() - t0) * 1000)
                top = np.argpartition(pr, -5)[-5:]
                preds.append(int(top[np.argmax(pr[top])]))
                t5 += wl.labels[i] in top
            preds = np.array(preds, dtype=np.int16)
            hashes.append(hashlib.sha256(preds.tobytes()).hexdigest()[:16])
            if r == 0:
                acc1, acc5 = float((preds == wl.labels).mean()), t5 / wl.n
        lat_sorted = sorted(lats)
        out["profiles"][name] = dict(
            res=res, rate_hz=p["rate_hz"], top1=round(acc1, 4), top5=round(acc5, 4),
            deterministic=len(set(hashes)) == 1, prediction_hashes=hashes,
            latency_p50_ms=round(statistics.median(lats), 2),
            latency_p95_ms=round(lat_sorted[int(0.95 * (len(lat_sorted) - 1))], 2),
            max_back_to_back_fps=round(1000 / statistics.fmean(lats), 1),
        )
        print(f"{name:17s} {res}px  top-1 {acc1:.1%}  top-5 {acc5:.1%}  "
              f"p50 {out['profiles'][name]['latency_p50_ms']} ms  deterministic={len(set(hashes)) == 1}")
    base = out["profiles"]["high_performance"]["top1"]
    for v in out["profiles"].values():
        v["top1_drop_vs_high_performance_pts"] = round((base - v["top1"]) * 100, 2)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"written {args.out}")


# ----------------------------------------------------------------------------- calibrate
def cmd_calibrate(args):
    """Set reduced_load rate_hz = fraction x balanced back-to-back FPS on THIS board."""
    cfg = load_profiles(args.profiles)
    wl = Workload(args, cfg)
    res = cfg["profiles"]["balanced"]["res"]
    wl.warmup(res, 20)
    n, idx, t0 = 0, 0, time.time()
    while time.time() - t0 < args.seconds:
        wl.infer(res, idx)
        idx = (idx + 1) % wl.n
        n += 1
    fps = n / (time.time() - t0)
    rate = round(args.fraction * fps, 1)
    print(f"balanced back-to-back: {fps:.1f} FPS -> reduced_load rate_hz = {rate}")
    if args.write:
        cfg["profiles"]["reduced_load"]["rate_hz"] = rate
        cfg["profiles"]["reduced_load"].pop("needs_rate", None)
        cfg["_calibrated"] = dict(utc=utc_iso(time.time()), host=os.uname().nodename,
                                  balanced_fps=round(fps, 1), fraction=args.fraction)
        with open(args.profiles, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"updated {args.profiles}")


# ----------------------------------------------------------------------------- cli
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--data", required=True, help="Imagenette val folder")
        p.add_argument("--models-dir", default=os.path.join(HERE, "models"))
        p.add_argument("--profiles", default=os.path.join(HERE, "profiles.json"))
        p.add_argument("--per-class", type=int, default=50)

    r = sub.add_parser("run", help="run the workload and log")
    common(r)
    r.add_argument("--profile", help="starting profile (default from profiles.json)")
    r.add_argument("--duration", type=float, default=0, help="seconds, 0 = until stopped")
    r.add_argument("--run-id")
    r.add_argument("--pcm-init", default="", help="initial PCM condition, e.g. solid")
    r.add_argument("--notes", default="")
    r.add_argument("--control", help="control file re-read every second")
    r.add_argument("--qos-file", help="QoS JSON rewritten every second")
    r.add_argument("--schedule", help='JSON: [{"t": 0, "profile": "high_performance"}, ...]')
    r.add_argument("--out-dir", default="logs")
    r.set_defaults(func=cmd_run)

    e = sub.add_parser("eval", help="offline accuracy per profile")
    common(e)
    e.add_argument("--repeats", type=int, default=2)
    e.add_argument("--out", default="profile_accuracy.json")
    e.set_defaults(func=cmd_eval)

    c = sub.add_parser("calibrate", help="set reduced_load rate on this board")
    common(c)
    c.add_argument("--seconds", type=float, default=30)
    c.add_argument("--fraction", type=float, required=True,
                   help="team-agreed fraction of balanced FPS for reduced_load (no default on purpose)")
    c.add_argument("--write", action="store_true", help="save the rate into profiles.json")
    c.set_defaults(func=cmd_calibrate)

    s = sub.add_parser("set-profile", help="request a profile change")
    s.add_argument("profile")
    s.add_argument("--control", required=True)
    s.set_defaults(func=lambda a: write_control(a.control, a.profile))

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
