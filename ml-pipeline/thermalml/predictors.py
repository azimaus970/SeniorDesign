"""Train, select, export and test the thermal predictors (Part III 4.3.3.1 B/C).

    python -m thermalml.predictors train  --config config/pipeline_config.yaml [--variants ...]
    python -m thermalml.predictors loro   --config ...      (optional stability check)
    python -m thermalml.predictors test   --config ...      (once, after selection)

train   fits every model, horizon, window and grid setting on the training runs,
        scores the validation runs, picks for each model and variant the setting
        with the lowest mean per-run validation RMSE at the primary horizon (Part III
        rule), and exports those packages. Both variants of a model use the same
        window W (Part III 4.3.3.2: M3 and M4 share the comparison window): W is the
        window whose best settings give the lowest validation RMSE averaged over the
        variants trained, and each variant then takes its best setting at that W.
loro    refits the selected settings leaving one train/validation run out at a time
        (fixed number of trees / epochs from the main fit) to show how stable they are.
test    scores the frozen packages on the test runs once and writes the paired
        temperature-only vs PCM-aware comparison.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics, splits
from .config import Config
from .features import Z_COLUMN, Samples, build_samples, steps
from .io import load_events, load_run, load_run_config
from .models import baselines, rnn, xgb
from .package import read_package, write_package

VARIANTS = ("temperature_only", "pcm_aware")


# ------------------------------------------------------------------ data
def load_runs(cfg: Config, run_ids: list[str], variant: str) -> dict:
    runs = {}
    for rid in run_ids:
        df = load_run(cfg.path_of("runs_dir") / f"{rid}.csv")
        if variant == "pcm_aware":
            zpath = cfg.path_of("results_dir") / "z_offline" / f"{rid}.csv"
            if not zpath.exists():
                raise FileNotFoundError(f"{zpath} missing: apply the selected estimator first "
                                        "(python -m thermalml.estimators_cli apply)")
            z = pd.read_csv(zpath)[["elapsed_s", Z_COLUMN]]
            df = df.merge(z, on="elapsed_s", how="left")
        ev = load_events(cfg.path_of("events_dir") / f"{rid}.csv", df)
        runs[rid] = (df, ev)
    return runs


def make_samples(runs: dict, cfg: Config, H, W, variant, with_seq, trend_s) -> list[Samples]:
    return [build_samples(df, cfg, H, W, variant, ev, with_seq, trend_s) for df, ev in runs.values()]


def common_origins(runs_by_variant: dict, cfg: Config, H, windows) -> dict:
    """Per run, the origins t that are eligible for every window in `windows` and every
    variant in `runs_by_variant` at horizon H. Compared errors (models, windows, M3/M4
    variants, baselines) are all scored on these same target instances; how many rows each
    setting could have used on its own is still reported in coverage.csv."""
    common: dict[str, set] = {}
    for variant, runs in runs_by_variant.items():
        for W in windows:
            for rid, (df, ev) in runs.items():
                t = set(np.round(build_samples(df, cfg, H, W, variant, ev, False, None).elapsed_s, 6).tolist())
                common[rid] = t if rid not in common else common[rid] & t
    return common


def restrict(samples: list[Samples], common: dict) -> list[Samples]:
    out = []
    for s in samples:
        keep = np.isin(np.round(s.elapsed_s, 6), np.fromiter(common.get(s.run_id, ()), float))
        idx = np.flatnonzero(keep)
        out.append(dataclasses.replace(
            s, elapsed_s=s.elapsed_s[idx], current_temp=s.current_temp[idx], y_delta=s.y_delta[idx],
            flat=s.flat.iloc[idx].reset_index(drop=True),
            seq=None if s.seq is None else s.seq[idx],
            recent_temp=None if s.recent_temp is None else s.recent_temp[idx],
            coverage=dict(s.coverage, scored_common_origins=int(len(idx)))))
    return out


def stack(samples: list[Samples], seq: bool = False):
    X = pd.concat([s.flat for s in samples], ignore_index=True)
    y = np.concatenate([s.y_delta for s in samples])
    S = None
    if seq:
        parts = [s.seq for s in samples if s.seq is not None]
        S = np.concatenate(parts) if parts else np.empty((0, 0, 0), np.float32)
    return X, y, S


def _split_by_run(samples: list[Samples], pred: np.ndarray):
    i = 0
    for s in samples:
        n = len(s.y_delta)
        yield s, pred[i:i + n]
        i += n


def _rows(samples, pred, model, variant, H, W, setting, split, band):
    return [dict(split=split, run_id=s.run_id, model=model, variant=variant, horizon_s=H, window_s=W,
                 setting=json.dumps(setting, sort_keys=True), y_true=s.y_delta, y_pred=p,
                 current_temp=s.current_temp, band_above_c=band)
            for s, p in _split_by_run(samples, pred)]


# ------------------------------------------------------------------ train
def train(cfg: Config, variants: list[str], models: list[str]):
    manifest = splits.load_manifest(cfg.path_of("split_manifest"))
    tr_ids, va_ids = splits.ids(manifest, "train"), splits.ids(manifest, "validation")
    period = float(cfg.require("timing.sampling_period_s"))
    horizons = cfg.require("predictor.horizons_s")
    windows = cfg.require("predictor.windows_s")
    primary = cfg.require("predictor.primary_horizon_s")
    if primary not in horizons:
        raise ValueError("predictor.primary_horizon_s must be one of predictor.horizons_s")
    trend_s = cfg.get("baselines.linear_trend_window_s")
    band = cfg.get("evaluation.report_band_above_c")
    need_seq = any(m in ("gru", "lstm") for m in models)
    results_dir = cfg.path_of("results_dir")
    rows, best = [], {}

    va_by_variant = {v: load_runs(cfg, va_ids, v) for v in variants}
    common = {H: common_origins(va_by_variant, cfg, H, windows) for H in horizons}
    for variant in variants:
        tr_runs, va_runs = load_runs(cfg, tr_ids, variant), va_by_variant[variant]
        for H in horizons:
            for W in windows:
                tw = trend_s if (trend_s is not None and trend_s <= W) else None
                s_tr = make_samples(tr_runs, cfg, H, W, variant, need_seq, tw)
                s_va = restrict(make_samples(va_runs, cfg, H, W, variant, need_seq, tw), common[H])
                X_tr, y_tr, S_tr = stack(s_tr, need_seq)
                X_va, y_va, S_va = stack(s_va, need_seq)
                if len(y_tr) == 0 or len(y_va) == 0:
                    raise ValueError(f"no eligible samples at H={H}, W={W} ({variant}); check coverage")
                cov = pd.DataFrame([dict(run_id=s.run_id, variant=variant, horizon_s=H, window_s=W, **s.coverage)
                                    for s in s_tr + s_va])
                cov.to_csv(results_dir / "coverage.csv", mode="a", index=False,
                           header=not (results_dir / "coverage.csv").exists())
                # baselines
                if variant == "temperature_only" and cfg.get("baselines.persistence", True):
                    rows += _rows(s_va, np.concatenate([baselines.persistence(s) for s in s_va]),
                                  "persistence", variant, H, W, {}, "validation", band)
                if variant == "temperature_only" and tw is not None:
                    rows += _rows(s_va, np.concatenate([baselines.linear_trend(s, period) for s in s_va]),
                                  "linear_trend", variant, H, W, {"window_s": tw}, "validation", band)
                # learned models
                for model in models:
                    if model == "xgboost":
                        settings = xgb.grid_settings(cfg.require("xgboost.grid"))
                    else:
                        settings = rnn.grid_settings(cfg.require("rnn.grid"))
                    for st in settings:
                        if model == "xgboost":
                            m = xgb.fit(X_tr, y_tr, X_va, y_va, st, int(cfg.require("xgboost.early_stopping_rounds")),
                                        int(cfg.get("xgboost.n_jobs", 1)))
                            p_va, info = xgb.predict(m, X_va), {"best_iteration": int(m.best_iteration)}
                        else:
                            m, info = rnn.train(model, S_tr, y_tr, S_va, y_va, st,
                                                float(cfg.require("rnn.learning_rate")),
                                                int(cfg.require("rnn.batch_size")), int(cfg.require("rnn.max_epochs")),
                                                int(cfg.require("rnn.early_stopping_patience")),
                                                int(cfg.require("rnn.seed")))
                            p_va = rnn.predict(m, S_va)
                        new = _rows(s_va, p_va, model, variant, H, W, st, "validation", band)
                        rows += new
                        score = float(np.mean([metrics.rmse(r["y_pred"] - r["y_true"]) for r in new]))
                        key = (model, variant, H, W)
                        if key not in best or score < best[key]["val_rmse_mean"]:
                            best[key] = dict(model_obj=m, setting=st, window_s=W, val_rmse_mean=score, info=info,
                                             features=list(X_tr.columns), seq_fields=s_tr[0].seq_fields,
                                             n_seq_features=None if S_tr is None else S_tr.shape[-1],
                                             X_va=X_va, S_va=S_va)

    per_run = metrics.per_run_table(rows)
    _append(per_run, results_dir / "evaluation_metrics.csv")
    metrics.summarize(per_run).to_csv(results_dir / "validation_summary.csv", index=False)

    # one shared window per model at the primary horizon, chosen symmetrically over the variants
    shared_w = {}
    for model in {k[0] for k in best}:
        per_w = {}
        for (m_, v_, H_, W_), b in best.items():
            if m_ == model and H_ == primary:
                per_w.setdefault(W_, {})[v_] = b["val_rmse_mean"]
        trained = {v for d_ in per_w.values() for v in d_}
        complete = {w: d_ for w, d_ in per_w.items() if set(d_) == trained}
        if complete:
            shared_w[model] = min(complete, key=lambda w: (float(np.mean(list(complete[w].values()))), w))
    selection = []
    for (model, variant, H, W), b in best.items():
        if H != primary or shared_w.get(model) != W:
            continue
        name = f"predictor_{model}_{variant}_H{H:g}"
        meta = _export(cfg, name, model, variant, H, b, manifest, tr_ids, va_ids)
        selection.append({k: meta[k] for k in ("package", "model", "variant", "horizon_s", "window_s",
                                               "setting", "val_rmse_mean")})
    pd.DataFrame(selection).to_csv(results_dir / "selection.csv", index=False)
    print(f"validation rows written to {results_dir/'evaluation_metrics.csv'}; selected packages in "
          f"{cfg.path_of('models_dir')}; model type is chosen by the team from both variants and Pi timing")


def _append(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, mode="a", index=False, header=not path.exists())


def _export(cfg, name, model, variant, H, b, manifest, tr_ids, va_ids) -> dict:
    models_dir = cfg.path_of("models_dir")
    period = float(cfg.require("timing.sampling_period_s"))
    meta = dict(kind="thermal_predictor", package=name, model=model, variant=variant, horizon_s=H,
                window_s=b["window_s"], window_steps=steps(b["window_s"], period, "window"),
                sampling_period_s=period, setting=b["setting"], training_info=b["info"],
                target="soc_temp_c change over H; predicted_temp_c = soc_temp_c + output",
                training_run_ids=tr_ids, validation_run_ids=va_ids,
                split_manifest_sha256=splits.manifest_hash(cfg.path_of("split_manifest")),
                val_rmse_mean=b["val_rmse_mean"], hold_max_periods=cfg.require("missing_data.hold_max_periods"),
                max_missing_fraction=cfg.require("missing_data.max_missing_fraction"))
    folder = Path(models_dir) / name
    if model == "xgboost":
        xgb.save(b["model_obj"], folder / "model.json")
        meta.update(model_file="model.json", feature_list=b["features"],
                    lags_s=cfg.require("predictor.lags_s"), slope_windows_s=cfg.require("predictor.slope_windows_s"))
    else:
        rnn.export_onnx(b["model_obj"], meta["window_steps"], b["n_seq_features"], folder / "model.onnx")
        m = b["model_obj"]
        meta.update(model_file="model.onnx", feature_list=b["seq_fields"],
                    scaler={"mean": m.x_mean.tolist(), "std": m.x_std.tolist(), "y_scale": float(m.y_scale)},
                    onnx_check=_onnx_check(m, folder / "model.onnx", b["S_va"]))
    write_package(models_dir, name, meta)
    return meta


def _onnx_check(model, path: Path, S_va: np.ndarray) -> dict:
    """Largest difference between PyTorch and ONNX Runtime outputs on the validation windows."""
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    a = rnn.predict(model, S_va)
    b = sess.run(None, {"window": S_va.astype(np.float32)})[0]
    return {"max_abs_diff_c": float(np.max(np.abs(a - b))), "n_windows": int(len(S_va))}


# ------------------------------------------------------------------ package inference (used by test and bench)
def load_predictor(folder: Path):
    meta = read_package(folder)
    if meta["model"] == "xgboost":
        m = xgb.load(Path(folder) / meta["model_file"])
        return meta, lambda s: xgb.predict(m, s.flat[meta["feature_list"]])
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(Path(folder) / meta["model_file"]), so, providers=["CPUExecutionProvider"])
    return meta, lambda s: sess.run(None, {"window": s.seq.astype(np.float32)})[0] if len(s.y_delta) else np.array([])


# ------------------------------------------------------------------ test
def test(cfg: Config, override_lock: bool):
    manifest = splits.load_manifest(cfg.path_of("split_manifest"))
    te_ids = splits.ids(manifest, "test")
    if not te_ids:
        raise ValueError("no test runs in the split manifest")
    sel = pd.read_csv(cfg.path_of("results_dir") / "selection.csv")
    lock = splits.TestLock(cfg.path_of("results_dir"), "predictors")
    lock.check_and_set({"packages": sel["package"].tolist(), "test_runs": te_ids,
                        "manifest_sha256": splits.manifest_hash(cfg.path_of("split_manifest"))}, override_lock)
    period = float(cfg.require("timing.sampling_period_s"))
    trend_s = cfg.get("baselines.linear_trend_window_s")
    band = cfg.get("evaluation.report_band_above_c")
    rows = []
    by_variant = {v: load_runs(cfg, te_ids, v) for v in sel["variant"].unique()}
    common = {float(H): common_origins(by_variant, cfg, float(H),
                                       sorted(set(sel.loc[sel["horizon_s"] == H, "window_s"])))
              for H in sel["horizon_s"].unique()}
    for variant in sel["variant"].unique():
        runs = by_variant[variant]
        for _, r in sel[sel["variant"] == variant].iterrows():
            meta, fn = load_predictor(cfg.path_of("models_dir") / r["package"])
            H, W = meta["horizon_s"], meta["window_s"]
            tw = trend_s if (trend_s is not None and trend_s <= W) else None
            ss = restrict(make_samples(runs, cfg, H, W, variant, meta["model"] != "xgboost", tw), common[float(H)])
            rows += _rows(ss, np.concatenate([fn(s) for s in ss]), meta["model"], variant, H, W,
                          meta["setting"], "test", band)
            if variant == "temperature_only":
                rows += _rows(ss, np.concatenate([baselines.persistence(s) for s in ss]),
                              "persistence", variant, H, W, {}, "test", band)
                if tw is not None:
                    rows += _rows(ss, np.concatenate([baselines.linear_trend(s, period) for s in ss]),
                                  "linear_trend", variant, H, W, {"window_s": tw}, "test", band)
    per_run = metrics.per_run_table(rows)
    out = cfg.path_of("results_dir")
    _append(per_run, out / "evaluation_metrics.csv")
    metrics.summarize(per_run).to_csv(out / "test_summary.csv", index=False)
    comps = []
    for model in sel["model"].unique():
        s = sel[sel["model"] == model]
        if set(s["variant"]) >= set(VARIANTS) and s["horizon_s"].nunique() == 1:
            H = float(s["horizon_s"].iloc[0])
            # both variants share the selected window (see train); pairing is by test run
            sub = per_run[(per_run.model == model) & (per_run.horizon_s == H)].assign(window_s=-1)
            if sub.loc[sub.variant == "pcm_aware", "run_id"].nunique() < 2:
                print(f"{model}: fewer than two test runs scored with both variants; no paired comparison")
                continue
            for metric in ("mae_c", "rmse_c"):
                c = metrics.compare_variants(
                    sub, model, H, -1, int(cfg.require("evaluation.bootstrap_resamples")),
                    float(cfg.require("evaluation.confidence_level")), int(cfg.require("evaluation.seed")), metric)
                c["window_s"] = "shared selected window"
                comps.append(c)
    if comps:
        pd.DataFrame(comps).to_csv(out / "variant_comparison.csv", index=False)
    print(f"test results in {out}")


# ------------------------------------------------------------------ leave-one-run-out
def loro(cfg: Config):
    manifest = splits.load_manifest(cfg.path_of("split_manifest"))
    sel = pd.read_csv(cfg.path_of("results_dir") / "selection.csv")
    out = []
    for _, r in sel.iterrows():
        meta = read_package(cfg.path_of("models_dir") / r["package"])
        H, W, variant, model = meta["horizon_s"], meta["window_s"], meta["variant"], meta["model"]
        pool = manifest.loc[manifest["split"].isin(["train", "validation"]), "run_id"].tolist()
        runs = load_runs(cfg, pool, variant)
        seq = model != "xgboost"
        samples = {rid: build_samples(runs[rid][0], cfg, H, W, variant, runs[rid][1], seq) for rid in pool}
        for fit_ids, held in splits.leave_one_out(manifest):
            X, y, S = stack([samples[i] for i in fit_ids], seq)
            s_h = samples[held]
            if model == "xgboost":
                import xgboost as xg
                st = dict(meta["setting"])
                st["n_estimators"] = int(meta["training_info"]["best_iteration"]) + 1
                m = xg.XGBRegressor(objective="reg:squarederror", n_jobs=1, **st).fit(X, y)
                p = m.predict(s_h.flat[list(X.columns)])
            else:
                epochs = int(np.argmin(meta["training_info"]["val_history"])) + 1
                m, _ = rnn.train(model, S, y, None, None, meta["setting"], float(cfg.require("rnn.learning_rate")),
                                 int(cfg.require("rnn.batch_size")), epochs, None, int(cfg.require("rnn.seed")))
                p = rnn.predict(m, s_h.seq)
            err = p - s_h.y_delta
            out.append(dict(package=r["package"], held_out_run=held, n=len(err),
                            mae_c=metrics.mae(err), rmse_c=metrics.rmse(err)))
    pd.DataFrame(out).to_csv(cfg.path_of("results_dir") / "loro_stability.csv", index=False)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["train", "test", "loro"])
    ap.add_argument("--config", default="config/pipeline_config.yaml")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    ap.add_argument("--models", nargs="+", default=["xgboost", "gru", "lstm"], choices=["xgboost", "gru", "lstm"])
    ap.add_argument("--override-test-lock", action="store_true")
    a = ap.parse_args(argv)
    cfg = Config.load(a.config)
    cfg.path_of("results_dir").mkdir(parents=True, exist_ok=True)
    if a.command == "train":
        train(cfg, a.variants, a.models)
    elif a.command == "test":
        test(cfg, a.override_test_lock)
    else:
        loro(cfg)


if __name__ == "__main__":
    main()
