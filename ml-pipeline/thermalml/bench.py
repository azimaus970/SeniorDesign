"""On-Pi call timing for exported packages (Part III 4.3.3.1 C, NFR7).

Run on the Raspberry Pi 5 with real windows taken from a logged run:

    python -m thermalml.bench --config config/pipeline_config.yaml --run data/runs/R001.csv \
        --packages predictor_xgboost_temperature_only_H20 estimator_E2 --calls N

N (number of timed calls) is chosen by whoever runs the benchmark. Each call
processes one window, as in one control cycle, with one thread. The median and
95th percentile are reported (Part III 4.3.3.1 C). Limits in the config are
checked only once the team has chosen which statistic they apply to.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from .config import Config
from .features import build_samples
from .io import load_run
from .package import read_package


def time_calls(fn, inputs, calls: int) -> np.ndarray:
    out = np.empty(calls)
    for i in range(calls):
        x = inputs[i % len(inputs)]
        t0 = time.perf_counter()
        fn(x)
        out[i] = (time.perf_counter() - t0) * 1000.0
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/pipeline_config.yaml")
    ap.add_argument("--run", required=True, help="a real logged run (Table 4.5 CSV)")
    ap.add_argument("--packages", nargs="+", required=True)
    ap.add_argument("--calls", type=int, required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    cfg = Config.load(a.config)
    run = load_run(a.run)
    rows = []
    for name in a.packages:
        folder = cfg.path_of("models_dir") / name
        meta = read_package(folder)
        if meta["kind"] == "thermal_predictor":
            from .predictors import load_predictor
            _, fn = load_predictor(folder)
            if meta["variant"] == "pcm_aware":
                raise SystemExit("time the pcm_aware package on a run that has z_hat_offline merged")
            s = build_samples(run, cfg, meta["horizon_s"], meta["window_s"], meta["variant"],
                              with_sequences=meta["model"] != "xgboost")
            if meta["model"] == "xgboost":
                inputs = [s.flat.iloc[[i]] for i in range(len(s.flat))]
                call = lambda X: fn(type("S", (), {"flat": X, "y_delta": [0]})())
            else:
                inputs = [s.seq[i:i + 1] for i in range(len(s.seq))]
                call = lambda W: fn(type("S", (), {"seq": W, "y_delta": [0]})())
        else:
            from .estimators.features import estimator_features
            from .estimators_cli import _load_estimator
            _, fn = _load_estimator(cfg, name)
            X = estimator_features(run, cfg).dropna()
            inputs = [X.iloc[[i]] for i in range(len(X))]
            call = fn
        if not inputs:
            raise SystemExit(f"{name}: no eligible windows in {a.run}")
        ms = time_calls(call, inputs, a.calls)
        rows.append({"package": name, "calls": a.calls, "p50_ms": float(np.median(ms)),
                     "p95_ms": float(np.percentile(ms, 95)), "max_ms": float(ms.max())})
    df = pd.DataFrame(rows)
    stat = cfg.get("runtime_limits.limit_statistic")
    if stat is not None:
        if stat not in ("p50", "p95"):
            raise SystemExit("runtime_limits.limit_statistic must be 'p50' or 'p95'")
        col = f"{stat}_ms"
        lim = cfg.get("runtime_limits.max_call_ms")
        if lim is not None:
            df["within_call_limit"] = df[col] <= float(lim)
    print(df.to_string(index=False))
    comb = cfg.get("runtime_limits.max_combined_ms")
    if len(df) > 1 and comb is not None:
        if stat is None:
            print(f"combined limit {comb} ms is set, but runtime_limits.limit_statistic is not; "
                  "reporting timings only")
        else:
            total = float(df[col].sum())
            print(json.dumps({"statistic": stat, "sum_ms": total, "combined_limit_ms": comb,
                              "within_combined_limit": bool(total <= float(comb))}))
    if a.out:
        df.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
