"""Error metrics (Part III Eq. 2.2) computed per run, then summarized across runs."""
from __future__ import annotations

import numpy as np
import pandas as pd


def mae(err: np.ndarray) -> float:
    return float(np.mean(np.abs(err))) if len(err) else float("nan")


def rmse(err: np.ndarray) -> float:
    return float(np.sqrt(np.mean(err ** 2))) if len(err) else float("nan")


def per_run_table(rows: list[dict]) -> pd.DataFrame:
    """rows: dicts with run_id, model, variant, horizon_s, window_s, y_true, y_pred, current_temp."""
    out = []
    for r in rows:
        err = np.asarray(r["y_pred"]) - np.asarray(r["y_true"])
        rec = {k: r[k] for k in ("split", "run_id", "model", "variant", "horizon_s", "window_s", "setting")}
        rec.update(n=len(err), mae_c=mae(err), rmse_c=rmse(err))
        band = r.get("band_above_c")
        if band is not None:
            sel = (np.asarray(r["current_temp"]) + np.asarray(r["y_true"])) >= band
            rec.update(band_above_c=band, n_band=int(sel.sum()), mae_band_c=mae(err[sel]), rmse_band_c=rmse(err[sel]))
        out.append(rec)
    return pd.DataFrame(out)


def summarize(per_run: pd.DataFrame) -> pd.DataFrame:
    """Mean and spread across runs, so one long run does not dominate (Part III 4.3.3.1 C)."""
    keys = ["split", "model", "variant", "horizon_s", "window_s", "setting"]
    g = per_run.groupby(keys, dropna=False)
    return g.agg(runs=("run_id", "nunique"), rows=("n", "sum"),
                 mae_mean=("mae_c", "mean"), mae_std=("mae_c", "std"),
                 rmse_mean=("rmse_c", "mean"), rmse_std=("rmse_c", "std")).reset_index()


def paired_bootstrap(diff: np.ndarray, resamples: int, level: float, seed: int) -> tuple[float, float, float]:
    """Mean of per-run differences with a percentile bootstrap interval over runs."""
    diff = np.asarray(diff, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(resamples, len(diff)))
    means = diff[idx].mean(axis=1)
    a = (1 - level) / 2
    return float(diff.mean()), float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def signflip_wilcoxon(diff: np.ndarray) -> float:
    """Two-sided exact p of the Wilcoxon signed-rank statistic, Pratt zero handling.

    Ranks of |d| are taken over all differences, zeros included (average ranks for
    ties); zero differences then contribute nothing, and the p value is the exact
    share of the 2^m sign assignments of the m non-zero differences whose statistic
    is at least as far from its centre as the observed one. With no zeros the
    smallest possible p is 2 / 2^n.
    """
    d = np.asarray(diff, float)
    a = np.abs(d)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a))
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    nz = d != 0
    r = ranks[nz]
    m = len(r)
    if m == 0:
        return 1.0
    if m > 20:
        raise ValueError("more than 20 non-zero differences: exact enumeration not implemented")
    centre = r.sum() / 2
    obs = abs(r[d[nz] > 0].sum() - centre)
    signs = ((np.arange(2 ** m)[:, None] >> np.arange(m)) & 1).astype(float)
    stats = np.abs(signs @ r - centre)
    return float(np.mean(stats >= obs - 1e-12))


def compare_variants(per_run: pd.DataFrame, model: str, horizon_s: float, window_s: float,
                     resamples: int, level: float, seed: int, metric: str = "mae_c") -> dict:
    """Run-by-run temperature-only minus PCM-aware error on the same runs (Part III 4.3.3.1 B).

    Positive values mean the PCM-aware (M4) variant has lower error. An improvement is
    claimed only if the whole interval is above zero and the exact sign-flip Wilcoxon
    test agrees (p < 1 - level), because a percentile interval over few runs can be
    too narrow to rely on alone.
    """
    sub = per_run[(per_run.model == model) & (per_run.horizon_s == horizon_s) & (per_run.window_s == window_s)]
    piv = sub.pivot_table(index="run_id", columns="variant", values=metric)
    piv = piv.dropna(subset=["temperature_only", "pcm_aware"])
    d = (piv["temperature_only"] - piv["pcm_aware"]).to_numpy()
    if len(d) < 2:
        raise ValueError("need at least two runs scored with both variants")
    mean, lo, hi = paired_bootstrap(d, resamples, level, seed)
    p = signflip_wilcoxon(d)
    return {"model": model, "horizon_s": horizon_s, "window_s": window_s, "metric": metric, "runs": len(d),
            "mean_diff": mean, "ci_low": lo, "ci_high": hi, "confidence_level": level,
            "per_run_spread_std": float(np.std(d, ddof=1)), "signflip_p": p,
            "improvement_claimed": bool(lo > 0 and d.mean() > 0 and p < 1 - level)}
