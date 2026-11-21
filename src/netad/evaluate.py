"""Overall ranking metrics + per-attack-family detection at a fixed false-alarm budget."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def threshold_at_fpr(normal_scores: np.ndarray, fpr: float = 0.05) -> float:
    """Alert when score > threshold; ~`fpr` of held-out NORMAL traffic then alerts."""
    return float(np.quantile(normal_scores, 1.0 - fpr))


def overall(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {"pr_auc": float(average_precision_score(y_true, scores)),
            "roc_auc": float(roc_auc_score(y_true, scores))}


def per_family_detection(families: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    """Detection rate per attack family; 'normal' row is the realised false-alarm rate."""
    out = {}
    for fam in sorted(set(families)):
        out[fam] = float((scores[families == fam] > threshold).mean())  # strict: ties at 0 are not alerts
    return out


def to_markdown(df: pd.DataFrame) -> str:
    cols = ["model"] + [c for c in df.columns if c != "model"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df[cols].iterrows():
        lines.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines)
