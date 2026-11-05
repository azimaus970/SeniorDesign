"""XGBoost regressor on the flat feature vector (Part III Table 4.6)."""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd


def grid_settings(grid: dict) -> list[dict]:
    keys = sorted(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*[grid[k] for k in keys])]


def fit(X_tr: pd.DataFrame, y_tr: np.ndarray, X_va: pd.DataFrame, y_va: np.ndarray, setting: dict,
        early_stopping_rounds: int, n_jobs: int = 1):
    import xgboost as xgb
    model = xgb.XGBRegressor(objective="reg:squarederror", n_jobs=n_jobs,
                             early_stopping_rounds=early_stopping_rounds, **setting)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return model


def predict(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict(X[model.get_booster().feature_names]) if len(X) else np.array([])


def save(model, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))


def load(path: Path, n_jobs: int = 1):
    import xgboost as xgb
    m = xgb.XGBRegressor(n_jobs=n_jobs)
    m.load_model(str(path))
    return m
