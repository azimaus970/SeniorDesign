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


def signflip_mean(diff: np.ndarray) -> float:
    """Two-sided exact sign-flip (paired permutation) p value for the mean difference.

    The statistic is the mean of d itself, so the test answers the same question as
    the reported estimate and cannot favour the other direction (unlike a signed-rank
    statistic, whose rank sum can lie on the opposite side of the mean). Zero
    differences contribute nothing to any sign assignment; the p value is the exact
    share of the 2^m sign assignments of the m non-zero differences whose |mean| is at
    least the observed |mean|. With no zeros the smallest possible p is 2 / 2^n; with
    all differences zero p = 1 (no difference).
    """
    d = np.asarray(diff, float)
    nz = d[d != 0]
    m = len(nz)
    if m == 0:
        return 1.0
    if m > 20:
        raise ValueError("more than 20 non-zero differences: exact enumeration not implemented")
    signs = 1 - 2 * ((np.arange(2 ** m)[:, None] >> np.arange(m)) & 1)
    stats = np.abs(signs @ nz)
    obs = abs(nz.sum())
    return float(np.mean(stats >= obs - 1e-9 * max(1.0, obs)))


def compare_variants(per_run: pd.DataFrame, model: str, horizon_s: float, window_s: float,
                     resamples: int, level: float, seed: int, metric: str = "mae_c") -> dict:
    """Run-by-run temperature-only minus PCM-aware error on the same runs (Part III 4.3.3.1 B).

    Positive values mean the PCM-aware (M4) variant has lower error. An improvement is
    claimed only if the whole interval is above zero, the mean is above zero and the
    exact sign-flip test of that mean agrees (p < 1 - level), because a percentile
    interval over few runs can be too narrow to rely on alone.
    """
    sub = per_run[(per_run.model == model) & (per_run.horizon_s == horizon_s) & (per_run.window_s == window_s)]
    piv = sub.pivot_table(index="run_id", columns="variant", values=metric)
    piv = piv.dropna(subset=["temperature_only", "pcm_aware"])
    d = (piv["temperature_only"] - piv["pcm_aware"]).to_numpy()
    if len(d) < 2:
        raise ValueError("need at least two runs scored with both variants")
    mean, lo, hi = paired_bootstrap(d, resamples, level, seed)
    p = signflip_mean(d)
    return {"model": model, "horizon_s": horizon_s, "window_s": window_s, "metric": metric, "runs": len(d),
            "mean_diff": mean, "ci_low": lo, "ci_high": hi, "confidence_level": level,
            "per_run_spread_std": float(np.std(d, ddof=1)), "signflip_p": p,
            "improvement_claimed": bool(lo > 0 and d.mean() > 0 and p < 1 - level)}
