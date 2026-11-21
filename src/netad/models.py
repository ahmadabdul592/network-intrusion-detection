"""Detectors for three data regimes. Convention: HIGHER score = MORE anomalous.

A  Novelty detection   - fit on NORMAL traffic only (no attack labels needed)
B  Few-label learning  - SelfTrainingClassifier: few labelled + many unlabelled flows
C  Fully supervised    - Random Forest on every label of the known attacks (reference)
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.svm import OneClassSVM

from .autoencoder import make_autoencoder


class NoveltyDetector:
    regime = "A: normal-only novelty"

    def __init__(self, name, est, max_fit_rows=None, seed=42):
        self.name, self.est, self.max_fit_rows, self.seed = name, est, max_fit_rows, seed

    def fit(self, X, y=None):
        if self.max_fit_rows and len(X) > self.max_fit_rows:
            X = X[np.random.default_rng(self.seed).choice(len(X), self.max_fit_rows, replace=False)]
        self.est.fit(X)
        return self

    def score(self, X):
        return -self.est.score_samples(X)


class PCAReconstruction:
    """Classical baseline: reconstruction error from the top principal components."""
    regime = "A: normal-only novelty"

    def __init__(self, variance: float = 0.95, seed: int = 42):
        self.name = "PCA reconstruction"
        self.pca = PCA(n_components=variance, svd_solver="full", random_state=seed)

    def fit(self, X, y=None):
        self.pca.fit(X)
        return self

    def score(self, X):
        rec = self.pca.inverse_transform(self.pca.transform(X))
        return ((X - rec) ** 2).mean(axis=1)


class AutoencoderDetector(NoveltyDetector):
    def score(self, X):
        return self.est.score_samples(X)


class SelfTrainingDetector:
    regime = "B: few-label self-training"

    def __init__(self, label_fraction: float = 0.02, seed: int = 42):
        self.name = f"Self-training RF ({label_fraction:.0%} labels)"
        self.label_fraction, self.seed = label_fraction, seed
        self.est = SelfTrainingClassifier(
            RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=seed),
            threshold=0.9, max_iter=10)

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        y_semi = np.full(len(y), -1)
        for cls in np.unique(y):                       # keep both classes in the labelled set
            idx = np.flatnonzero(y == cls)
            keep = rng.choice(idx, max(int(len(idx) * self.label_fraction), 5), replace=False)
            y_semi[keep] = y[keep]
        self.est.fit(X, y_semi)
        return self

    def score(self, X):
        return self.est.predict_proba(X)[:, 1]


class SupervisedRF:
    regime = "C: fully supervised"

    def __init__(self, seed: int = 42):
        self.name = "Supervised RF (known attacks)"
        self.est = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=seed)

    def fit(self, X, y):
        self.est.fit(X, y)
        return self

    def score(self, X):
        return self.est.predict_proba(X)[:, 1]


def build_models(seed: int = 42, backend: str = "auto", label_fraction: float = 0.02):
    ae = make_autoencoder(backend, seed)
    return [
        NoveltyDetector("Isolation Forest", IsolationForest(n_estimators=300, random_state=seed, n_jobs=-1), seed=seed),
        NoveltyDetector("One-Class SVM", OneClassSVM(kernel="rbf", gamma="scale", nu=0.05),
                        max_fit_rows=10_000, seed=seed),
        PCAReconstruction(seed=seed),
        AutoencoderDetector(ae.name, ae, seed=seed),
        SelfTrainingDetector(label_fraction, seed),
        SupervisedRF(seed),
    ]
