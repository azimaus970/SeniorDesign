#!/usr/bin/env python3
"""
merge_logs.py - fill the AI fields of the Table 4.5 main CSV from the AI sidecar.

Causal join (no future information): for each main-CSV row with timestamp T,
the sidecar row of the same run_id used is the one PUBLISHED most recently at or
before T (published_t_unix: after the window's QoS was published, so the
controller could have read it), and only if it was published less than --tol
seconds before T (default 1.0 s, the Part III control period). This is the
"preceding-interval QoS" of Part III 4.3.3.2. Both logs use the shared Pi clock (UTC).
  * A window published after T is never used, even if the window itself ended before T.
  * Logs written before published_t_unix existed fall back to the window end
    (t_unix), with a warning.
  * Two sidecar rows of one run with the same publication time are an error, not averaged.
  * Rows are never matched across runs.
  * A main row with no window ending within --tol before it gets empty values, never 0.
Filled columns:
    workload_profile      <- sidecar workload_profile (profile actually running)
    inference_fps         <- sidecar inference_fps
    inference_latency_ms  <- sidecar inference_latency_ms
Existing values in the main CSV are kept unless --overwrite is given. Accuracy is not copied.

Usage:
    python merge_logs.py main.csv R001_ai.csv -o main_merged.csv
"""
import argparse
import bisect
import csv
import datetime as dt
import sys

MAP = {"workload_profile": "workload_profile",
       "inference_fps": "inference_fps",
       "inference_latency_ms": "inference_latency_ms"}


def to_unix(ts):
    t = dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if t.tzinfo is None:
        raise ValueError(f"timestamp without timezone: {ts!r} (the Pi clock must be logged in UTC)")
    return t.timestamp()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("main_csv")
    ap.add_argument("ai_csv", nargs="+")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--time-col", default="timestamp", help="ISO 8601 column in the main CSV")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--tol", type=float, default=1.0,
                    help="max age (s) of the AI window's publication before the main-row time (Part III control period)")
    args = ap.parse_args()

    side = {}
    for path in args.ai_csv:
        with open(path) as f:
            for r in csv.DictReader(f):
                key = r.get("published_t_unix") or ""
                if not key:
                    print(f"warning: {path} has no published_t_unix; using window end t_unix", file=sys.stderr)
                    key = r["t_unix"]
                side.setdefault(r["run_id"], []).append((float(key), r))
    for rid, lst in side.items():
        lst.sort(key=lambda x: x[0])
        for (t0, _), (t1, _) in zip(lst, lst[1:]):
            if t1 == t0:
                raise ValueError(f"run {rid}: two AI-log rows published at {t0}")
    ends = {rid: [t for t, _ in lst] for rid, lst in side.items()}

    with open(args.main_csv) as f:
        rd = csv.DictReader(f)
        fields = list(rd.fieldnames)
        for c in MAP:
            if c not in fields:
                fields.append(c)
        out_rows, hit, miss = [], 0, 0
        for r in rd:
            T = to_unix(r[args.time_col])
            s = None
            if r["run_id"] in side:
                i = bisect.bisect_right(ends[r["run_id"]], T) - 1   # last window published at or before T
                if i >= 0 and T - ends[r["run_id"]][i] < args.tol:
                    s = side[r["run_id"]][i][1]
            for dst, src in MAP.items():
                if args.overwrite or not r.get(dst):
                    r[dst] = s[src] if s else ""
            hit += s is not None
            miss += s is None
            out_rows.append(r)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)
    print(f"{hit} rows matched, {miss} rows without AI data (left empty) -> {args.out}")


if __name__ == "__main__":
    main()
