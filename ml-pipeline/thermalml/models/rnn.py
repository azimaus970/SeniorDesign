"""GRU and LSTM predictors on the W x F window (Part III Table 4.6, 4.3.3.1 B).

Training settings (grid, learning rate, batch size, epochs, patience, seed) all come
from the config. Scaling parameters are fitted on training runs only and are baked
into the exported ONNX graph, so the Pi feeds raw, held values in the logged units
and receives the predicted change in degrees C.
"""
from __future__ import annotations

import copy
import itertools
from pathlib import Path

import numpy as np


def grid_settings(grid: dict) -> list[dict]:
    keys = sorted(grid)
    return [dict(zip(keys, v)) for v in itertools.product(*[grid[k] for k in keys])]


def fit_scaler(seq_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = seq_train.reshape(-1, seq_train.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std == 0] = 1.0           # a constant feature carries no information; avoid division by zero
    return mean.astype(np.float32), std.astype(np.float32)


def _torch():
    import torch
    import torch.nn as nn
    return torch, nn


def build(cell: str, n_features: int, hidden: int, layers: int, dropout: float,
          x_mean: np.ndarray, x_std: np.ndarray, y_scale: float):
    torch, nn = _torch()

    class SeqRegressor(nn.Module):
        def __init__(self):
            super().__init__()
            rnn_cls = {"gru": nn.GRU, "lstm": nn.LSTM}[cell]
            self.rnn = rnn_cls(n_features, hidden, num_layers=layers, batch_first=True,
                               dropout=dropout if layers > 1 else 0.0)
            self.head = nn.Linear(hidden, 1)
            self.register_buffer("x_mean", torch.tensor(x_mean))
            self.register_buffer("x_std", torch.tensor(x_std))
            self.register_buffer("y_scale", torch.tensor(float(y_scale)))

        def forward_scaled(self, x):            # x already normalized
            out, _ = self.rnn(x)
            return self.head(out[:, -1, :]).squeeze(-1)

        def forward(self, x_raw):               # raw window -> change in degrees C
            x = (x_raw - self.x_mean) / self.x_std
            return self.forward_scaled(x) * self.y_scale

    return SeqRegressor()


def train(cell: str, seq_tr, y_tr, seq_va, y_va, setting: dict, lr: float, batch_size: int,
          max_epochs: int, patience: int, seed: int):
    torch, nn = _torch()
    torch.manual_seed(seed)
    np.random.seed(seed)
    x_mean, x_std = fit_scaler(seq_tr)
    y_scale = float(np.std(y_tr)) or 1.0
    model = build(cell, seq_tr.shape[-1], int(setting["hidden_units"]), int(setting["layers"]),
                  float(setting.get("dropout", 0.0)), x_mean, x_std, y_scale)
    xt = torch.tensor((seq_tr - x_mean) / x_std, dtype=torch.float32)
    yt = torch.tensor(y_tr / y_scale, dtype=torch.float32)
    fixed = seq_va is None          # fixed number of epochs, no validation (leave-one-run-out refit)
    if not fixed:
        xv = torch.tensor((seq_va - x_mean) / x_std, dtype=torch.float32)
        yv = torch.tensor(y_va / y_scale, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best, best_state, bad, history = float("inf"), None, 0, []
    gen = torch.Generator().manual_seed(seed)
    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(len(xt), generator=gen)
        for i in range(0, len(xt), batch_size):
            b = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model.forward_scaled(xt[b]), yt[b])
            loss.backward()
            opt.step()
        model.eval()
        if fixed:
            continue
        with torch.no_grad():
            val = float(loss_fn(model.forward_scaled(xv), yv))
        history.append(val)
        if val < best:
            best, best_state, bad = val, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    if fixed:
        model.eval()
        return model, {"epochs_run": max_epochs}
    model.load_state_dict(best_state)
    model.eval()
    return model, {"epochs_run": len(history), "best_val_mse_scaled": best, "val_history": history}


def predict(model, seq: np.ndarray) -> np.ndarray:
    torch, _ = _torch()
    if len(seq) == 0:
        return np.array([])
    with torch.no_grad():
        return model(torch.tensor(seq, dtype=torch.float32)).numpy()


def export_onnx(model, window_steps: int, n_features: int, path: Path):
    torch, _ = _torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, window_steps, n_features)
    torch.onnx.export(model, (dummy,), str(path), input_names=["window"], output_names=["delta_c"],
                      dynamic_axes={"window": {0: "batch"}, "delta_c": {0: "batch"}}, opset_version=17,
                      dynamo=False)
