import numpy as np
import pytest

from netad.data import FEATURES, build_preprocessor, make_synthetic
from netad.evaluate import per_family_detection, threshold_at_fpr
from netad.train import run


def test_synthetic_schema_and_novel_family():
    train, test = make_synthetic(seed=1, scale=0.05)
    assert set(FEATURES) <= set(train.columns)
    assert "r2l" in set(test["family"]) and "r2l" not in set(train["family"])
    assert train["is_attack"].isin([0, 1]).all()


def test_preprocessor_shape_and_unknown_category():
    train, test = make_synthetic(seed=2, scale=0.05)
    pre = build_preprocessor().fit(train[FEATURES])
    Z = pre.transform(test[FEATURES])
    assert np.isfinite(Z).all() and Z.shape[0] == len(test)
    odd = test[FEATURES].head(3).copy()
    odd["service"] = "never_seen_service"          # must not crash
    assert pre.transform(odd).shape[0] == 3


def test_threshold_gives_requested_false_alarm_rate():
    rng = np.random.default_rng(0)
    normal = rng.normal(size=20_000)
    thr = threshold_at_fpr(normal, 0.05)
    assert (normal > thr).mean() == pytest.approx(0.05, abs=0.005)
    fam = per_family_detection(np.array(["normal"] * 3 + ["dos"] * 2), np.array([0, 0, 9, 9, 9.]), 5.0)
    # saturated classifier (all-zero normal scores): threshold 0 must not alert on ties
    assert per_family_detection(np.array(["normal"] * 4), np.zeros(4), 0.0)["normal"] == 0.0
    assert fam["dos"] == 1.0 and fam["normal"] == pytest.approx(1 / 3)


def test_end_to_end(tmp_path):
    res, fam = run(backend="sklearn", out=str(tmp_path), scale=0.15, save=True)
    assert (tmp_path / "metrics.md").exists() and (tmp_path / "per_family_detection.png").exists()
    r = res.set_index("model")
    assert r.loc["Isolation Forest", "roc_auc"] > 0.7
    assert (r["roc_auc"].between(0, 1)).all()
    assert any("NOVEL" in c for c in fam.columns)
