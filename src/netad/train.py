"""Benchmark:  python -m netad.train [--data-dir data] [--backend auto|torch|sklearn]"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .data import build_preprocessor, load_data, FEATURES
from .evaluate import overall, per_family_detection, threshold_at_fpr, to_markdown
from .models import SelfTrainingDetector, SupervisedRF, build_models


def run(data_dir: str = "data", backend: str = "auto", out: str = "results", seed: int = 42,
        scale: float = 1.0, label_fraction: float = 0.02, fpr: float = 0.05,
        save: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, test, source = load_data(data_dir, seed, scale)

    # Hold out 15% of NORMAL training flows to calibrate alert thresholds (never used to fit).
    normal_idx = train.index[train["is_attack"] == 0]
    fit_normal_idx, val_idx = train_test_split(normal_idx, test_size=0.15, random_state=seed)
    attack_idx = train.index[train["is_attack"] == 1]

    pre = build_preprocessor().fit(train.loc[fit_normal_idx, FEATURES])   # scaler sees normal only
    Z = lambda df: pre.transform(df[FEATURES])  # noqa: E731
    Z_normal, Z_val, Z_test = Z(train.loc[fit_normal_idx]), Z(train.loc[val_idx]), Z(test)
    Z_mixed = Z(train.loc[np.concatenate([fit_normal_idx, attack_idx])])
    y_mixed = train.loc[np.concatenate([fit_normal_idx, attack_idx]), "is_attack"].to_numpy()
    y_test, fam_test = test["is_attack"].to_numpy(), test["family"].to_numpy()
    train_families = sorted(set(train["family"]) - {"normal"})
    novel = sorted(set(test["family"]) - set(train["family"]) - {"normal"})
    print(f"[data] {source}: train={len(train):,} test={len(test):,}  "
          f"known attacks={train_families}  NOVEL in test={novel}")

    overall_rows, family_rows, roc_data = [], [], {}
    for m in build_models(seed, backend, label_fraction):
        t0 = time.perf_counter()
        if isinstance(m, (SelfTrainingDetector, SupervisedRF)):
            m.fit(Z_mixed, y_mixed)
        else:
            m.fit(Z_normal)
        fit_s = time.perf_counter() - t0

        s_test, s_val = m.score(Z_test), m.score(Z_val)
        thr = threshold_at_fpr(s_val, fpr)
        o = overall(y_test, s_test)
        fam = per_family_detection(fam_test, s_test, thr)
        overall_rows.append({"model": m.name, "regime": m.regime, **o, "fit_s": fit_s})
        family_rows.append({"model": m.name, **{
            (f"{k} (NOVEL)" if k in novel else k if k != "normal" else "false_alarm_rate"): v
            for k, v in fam.items()}})
        roc_data[m.name] = (y_test, s_test)
        print(f"[done] {m.name:<34} PR-AUC={o['pr_auc']:.3f}  ROC-AUC={o['roc_auc']:.3f}  "
              + "  ".join(f"{k}={v:.2f}" for k, v in fam.items()))

    res = pd.DataFrame(overall_rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)
    fam_df = pd.DataFrame(family_rows).set_index("model").loc[res["model"]].reset_index()

    if save:
        d = Path(out)
        d.mkdir(parents=True, exist_ok=True)
        res.to_csv(d / "metrics.csv", index=False)
        fam_df.to_csv(d / "per_family.csv", index=False)
        (d / "metrics.md").write_text(
            f"### Overall (test set incl. novel attacks)\n\n{to_markdown(res)}\n\n"
            f"### Detection rate per attack family at {fpr:.0%} false-alarm budget\n\n{to_markdown(fam_df)}\n")
        (d / "run_config.json").write_text(json.dumps(
            {"source": source, "backend": backend, "seed": seed, "label_fraction": label_fraction,
             "fpr_budget": fpr, "known_attacks": train_families, "novel_attacks": novel}, indent=2))
        _plot(fam_df, novel, d / "per_family_detection.png", fpr)
        print(f"[saved] {d}/metrics.csv, per_family.csv, metrics.md, per_family_detection.png")
    return res, fam_df


def _plot(fam_df: pd.DataFrame, novel: list[str], path: Path, fpr: float) -> None:
    cols = [c for c in fam_df.columns if c not in ("model", "false_alarm_rate")]
    ax = fam_df.set_index("model")[cols].plot(kind="bar", figsize=(9, 5), width=0.8)
    ax.set(ylabel="Detection rate", ylim=(0, 1.05),
           title=f"Detection by attack family @ {fpr:.0%} false alarms")
    plt.xticks(rotation=25, ha="right", fontsize=8)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--backend", choices=["auto", "torch", "sklearn"], default="auto")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scale", type=float, default=1.0, help="synthetic data size multiplier")
    ap.add_argument("--label-fraction", type=float, default=0.02)
    ap.add_argument("--fpr", type=float, default=0.05, help="false-alarm budget for per-family detection")
    a = ap.parse_args()
    res, fam = run(a.data_dir, a.backend, a.out, a.seed, a.scale, a.label_fraction, a.fpr)
    print("\n" + to_markdown(res) + "\n\n" + to_markdown(fam))


if __name__ == "__main__":
    main()
