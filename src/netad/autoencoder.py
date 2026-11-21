"""Autoencoder anomaly detectors. Anomaly score = reconstruction error (MSE).

Two interchangeable backends:
  * TorchAutoencoder   - PyTorch (preferred; used when torch is installed)
  * SklearnAutoencoder - MLPRegressor trained to reproduce its input
                         (dependency-light fallback, used in CI)
"""
from __future__ import annotations

import copy

import numpy as np


class SklearnAutoencoder:
    name = "Autoencoder (sklearn)"

    def __init__(self, hidden: tuple[int, ...] = (16, 8, 16), max_iter: int = 80,
                 seed: int = 42) -> None:
        self.hidden, self.max_iter, self.seed = hidden, max_iter, seed
        self.model = None

    def fit(self, X: np.ndarray) -> "SklearnAutoencoder":
        from sklearn.neural_network import MLPRegressor
        self.model = MLPRegressor(
            hidden_layer_sizes=self.hidden, activation="relu", solver="adam",
            max_iter=self.max_iter, early_stopping=True, n_iter_no_change=5,
            random_state=self.seed,
        )
        self.model.fit(X, X)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.model.predict(X)) ** 2).mean(axis=1)


class TorchAutoencoder:
    name = "Autoencoder (PyTorch)"

    def __init__(self, hidden: tuple[int, ...] = (16, 8), epochs: int = 40,
                 batch_size: int = 512, lr: float = 1e-3, weight_decay: float = 1e-5,
                 patience: int = 5, val_fraction: float = 0.1, seed: int = 42,
                 device: str | None = None) -> None:
        self.hidden, self.epochs, self.batch_size = hidden, epochs, batch_size
        self.lr, self.weight_decay, self.patience = lr, weight_decay, patience
        self.val_fraction, self.seed, self.device = val_fraction, seed, device
        self.model = None
        self.history: list[tuple[float, float]] = []

    def _build(self, n_in: int):
        from torch import nn
        dims = [n_in, *self.hidden]
        enc: list = []
        for a, b in zip(dims[:-1], dims[1:]):
            enc += [nn.Linear(a, b), nn.ReLU()]
        rdims = dims[::-1]
        dec: list = []
        for i, (a, b) in enumerate(zip(rdims[:-1], rdims[1:])):
            dec.append(nn.Linear(a, b))
            if i < len(rdims) - 2:
                dec.append(nn.ReLU())
        return nn.Sequential(*enc, *dec)

    def fit(self, X: np.ndarray) -> "TorchAutoencoder":
        import torch
        torch.manual_seed(self.seed)
        dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._dev = dev
        X = np.asarray(X, dtype=np.float32)
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(len(X))
        n_val = max(int(len(X) * self.val_fraction), 1)
        val = torch.from_numpy(X[perm[:n_val]]).to(dev)
        train = torch.from_numpy(X[perm[n_val:]]).to(dev)

        self.model = self._build(X.shape[1]).to(dev)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)
        loss_fn = torch.nn.MSELoss()
        best, best_state, bad = float("inf"), None, 0

        for _ in range(self.epochs):
            self.model.train()
            idx = torch.randperm(len(train), device=dev)
            running = 0.0
            for s in range(0, len(train), self.batch_size):
                batch = train[idx[s:s + self.batch_size]]
                opt.zero_grad()
                loss = loss_fn(self.model(batch), batch)
                loss.backward()
                opt.step()
                running += loss.item() * len(batch)
            self.model.eval()
            with torch.no_grad():
                val_loss = loss_fn(self.model(val), val).item()
            self.history.append((running / len(train), val_loss))
            if val_loss < best - 1e-6:
                best, best_state, bad = val_loss, copy.deepcopy(self.model.state_dict()), 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        import torch
        X = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(self._dev)
        out = []
        with torch.no_grad():
            for s in range(0, len(X), 4096):
                b = X[s:s + 4096]
                out.append(((self.model(b) - b) ** 2).mean(dim=1).cpu().numpy())
        return np.concatenate(out)


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def make_autoencoder(backend: str = "auto", seed: int = 42):
    if backend == "auto":
        backend = "torch" if torch_available() else "sklearn"
    if backend == "torch":
        return TorchAutoencoder(seed=seed)
    if backend == "sklearn":
        return SklearnAutoencoder(seed=seed)
    raise ValueError(f"Unknown backend: {backend!r} (use auto|torch|sklearn)")
