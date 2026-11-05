"""E1 (tabular) and E2 (stateful sequence) estimators, trained on heater cycles
(Part III 4.3.3.1 A). Reference z comes from calibration.reference_z.
"""
from __future__ import annotations

import copy
import itertools
from pathlib import Path

import numpy as np
import pandas as pd


def grid_settings(grid: dict) -> list[dict]:
    keys = sorted(grid)
    return [dict(zip(keys, v)) for v in itertools.product(*[grid[k] for k in keys])]


# ---------------------------------------------------------------- E1
def fit_e1(kind: str, X_tr, y_tr, X_va, y_va, setting: dict, early_stopping_rounds: int | None = None):
    """kind: 'xgboost' or 'ridge' (Part III: XGBoost or a regularized linear model)."""
    if kind == "xgboost":
        import xgboost as xgb
        m = xgb.XGBRegressor(objective="reg:squarederror", n_jobs=1,
                             early_stopping_rounds=early_stopping_rounds, **setting)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        return m
    if kind == "ridge":
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(), Ridge(**setting)).fit(X_tr, y_tr)
    raise ValueError("estimator.e1.model must be 'xgboost' or 'ridge'")


def predict_e1(model, X: pd.DataFrame) -> np.ndarray:
    return np.clip(model.predict(X), 0.0, 1.0)


# ---------------------------------------------------------------- E2
def _torch():
    import torch
    import torch.nn as nn
    return torch, nn


def build_e2(cell: str, n_features: int, hidden: int, x_mean, x_std):
    torch, nn = _torch()

    class StatefulEstimator(nn.Module):
        """One step per call; the hidden state is an explicit input and output, because
        ONNX Runtime keeps no state between calls (onnxruntime issue 11085)."""

        def __init__(self):
            super().__init__()
            self.cell_type = cell
            self.rnn = {"gru": nn.GRU, "lstm": nn.LSTM}[cell](n_features, hidden, batch_first=True)
            self.head = nn.Linear(hidden, 1)
            self.hidden = hidden
            self.register_buffer("x_mean", torch.tensor(np.asarray(x_mean, np.float32)))
            self.register_buffer("x_std", torch.tensor(np.asarray(x_std, np.float32)))

        def zero_state(self, batch: int):
            h = torch.zeros(1, batch, self.hidden)
            return (h, torch.zeros_like(h)) if self.cell_type == "lstm" else h

        def forward_seq(self, x_raw, state):
            x = (x_raw - self.x_mean) / self.x_std
            out, state = self.rnn(x, state)
            return torch.sigmoid(self.head(out)).squeeze(-1), state   # z in (0, 1)

    return StatefulEstimator()


def export_e2_onnx(model, n_features: int, path: Path):
    """Exports a single-step graph: inputs x (1,1,F), h_in [, c_in]; outputs z, h_out [, c_out]."""
    torch, nn = _torch()
    path.parent.mkdir(parents=True, exist_ok=True)

    class Step(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x, h, c=None):
            state = (h, c) if self.m.cell_type == "lstm" else h
            z, s = self.m.forward_seq(x, state)
            return (z, s[0], s[1]) if self.m.cell_type == "lstm" else (z, s)

    step = Step(model).eval()
    x = torch.zeros(1, 1, n_features)
    h = torch.zeros(1, 1, model.hidden)
    if model.cell_type == "lstm":
        args, ins, outs = (x, h, torch.zeros_like(h)), ["x", "h_in", "c_in"], ["z", "h_out", "c_out"]
    else:
        args, ins, outs = (x, h), ["x", "h_in"], ["z", "h_out"]
    torch.onnx.export(step, args, str(path), input_names=ins, output_names=outs, opset_version=17, dynamo=False)


def train_e2(cell: str, cycles_tr: list[tuple[np.ndarray, np.ndarray]],
             cycles_va: list[tuple[np.ndarray, np.ndarray]], hidden: int, chunk: int, lr: float,
             max_epochs: int, patience: int, seed: int):
    """Truncated backpropagation through each whole cycle, state carried between chunks.

    cycles_*: list of (X (T, F), z_ref (T,)) per heater cycle, with no missing values.
    """
    torch, nn = _torch()
    torch.manual_seed(seed)
    allx = np.concatenate([x for x, _ in cycles_tr])
    x_mean = allx.mean(axis=0)
    x_std = allx.std(axis=0)
    x_std[x_std == 0] = 1.0
    model = build_e2(cell, allx.shape[1], hidden, x_mean, x_std)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    def run_cycle(x, train: bool):
        state = model.zero_state(1)
        xt = torch.tensor(x[None], dtype=torch.float32)
        preds = []
        for i in range(0, xt.shape[1], chunk):
            z, state = model.forward_seq(xt[:, i:i + chunk], state)
            preds.append(z)
            state = tuple(s.detach() for s in state) if isinstance(state, tuple) else state.detach()
            if train:
                yield z, i
        if not train:
            yield torch.cat(preds, dim=1), None

    def val_loss():
        model.eval()
        with torch.no_grad():
            errs = []
            for x, z in cycles_va:
                pred, _ = next(run_cycle(x, False))
                errs.append(float(loss_fn(pred[0], torch.tensor(z, dtype=torch.float32))))
        return float(np.mean(errs))

    best, best_state, bad, history = float("inf"), None, 0, []
    rng = np.random.default_rng(seed)
    for epoch in range(max_epochs):
        model.train()
        for ci in rng.permutation(len(cycles_tr)):
            x, z = cycles_tr[ci]
            zt = torch.tensor(z, dtype=torch.float32)
            for pred, i in run_cycle(x, True):
                opt.zero_grad()
                loss = loss_fn(pred[0], zt[i:i + pred.shape[1]])
                loss.backward()
                opt.step()
        v = val_loss()
        history.append(v)
        if v < best:
            best, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, {"epochs_run": len(history), "best_val_mse": best, "val_history": history}


def predict_e2(model, x: np.ndarray, reset_steps: np.ndarray | None = None) -> np.ndarray:
    """Run a whole cycle or run step by step. reset_steps: indices where the state is reset
    (the endpoint-reset variant of Part IV 5.2.4.2)."""
    torch, _ = _torch()
    resets = set(map(int, reset_steps)) if reset_steps is not None else set()
    state = model.zero_state(1)
    out = np.empty(len(x))
    with torch.no_grad():
        for i in range(len(x)):
            if i in resets:
                state = model.zero_state(1)
            if not np.all(np.isfinite(x[i])):
                out[i] = np.nan          # missing input: no output, state not advanced
                continue
            z, state = model.forward_seq(torch.tensor(x[None, i:i + 1], dtype=torch.float32), state)
            out[i] = float(z[0, 0])
    return out
