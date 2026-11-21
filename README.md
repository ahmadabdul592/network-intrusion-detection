# Network Intrusion Detection: Semi-Supervised & Novelty Detection

Can a model flag **attacks it has never seen**? This project compares normal-traffic-only novelty detectors,
a few-label semi-supervised classifier and a fully supervised model on NSL-KDD-style network flows, and reports
**detection rate per attack family** at a fixed false-alarm budget, which is what a SOC analyst cares about.

![CI](https://github.com/ahmadabdul592/network-intrusion-detection/actions/workflows/ci.yml/badge.svg)

## Setup
| Regime | Models | Labels used |
|---|---|---|
| **A. Novelty** | Isolation Forest, One-Class SVM, PCA reconstruction, Autoencoder (PyTorch / sklearn fallback) | none, trained on normal traffic only |
| **B. Semi-supervised** | `SelfTrainingClassifier(RandomForest)` | 2% of labels (configurable), the rest unlabelled |
| **C. Supervised reference** | Random Forest | all labels of the *known* attacks |

- **Train/test shift by design:** the test set contains an attack family absent from training.
- **Threshold calibration:** 15% of normal training flows are held out and the alert threshold is set so ~5% of normal traffic alerts. The same false-alarm budget is applied to every model, so the comparison is fair.
- **No leakage:** the preprocessor (log1p + scaling + one-hot with unknown-category handling) is fit on normal training traffic only.

## Quick start
```bash
pip install -e ".[dev]"      # add ".[torch]" for the PyTorch autoencoder
pytest -q
python -m netad.train
```
For real data, put `KDDTrain+.txt` and `KDDTest+.txt` ([NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html)) in `data/`.
Attack names are mapped to DoS / Probe / R2L / U2R families automatically. With no files present, a synthetic generator with the same 41 features is used.

## Results

Detection rate per family at a ~5% false-alarm budget (R2L is unseen in training):

NSL-KDD: train=125,973 test=22,544  known attacks=['dos', 'probe', 'r2l', 'u2r']  NOVEL in test=[]
Isolation Forest                   PR-AUC=0.961  ROC-AUC=0.951  dos=0.77  normal=0.02  probe=0.97  r2l=0.09  u2r=0.33
One-Class SVM                      PR-AUC=0.945  ROC-AUC=0.910  dos=0.91  normal=0.08  probe=0.94  r2l=0.44  u2r=0.76
PCA reconstruction                 PR-AUC=0.949  ROC-AUC=0.958  dos=0.83  normal=0.04  probe=0.69  r2l=0.32  u2r=0.63
Autoencoder (PyTorch)              PR-AUC=0.951  ROC-AUC=0.958  dos=0.85  normal=0.04  probe=0.78  r2l=0.35  u2r=0.69
Self-training RF (2% labels)       PR-AUC=0.953  ROC-AUC=0.948  dos=0.99  normal=0.14  probe=1.00  r2l=0.86  u2r=0.66
Supervised RF (known attacks)      PR-AUC=0.964  ROC-AUC=0.962  dos=1.00  normal=0.13  probe=1.00  r2l=0.90  u2r=0.94

| model | regime | pr_auc | roc_auc | fit_s |
|---|---|---|---|---|
| Supervised RF (known attacks) | C: fully supervised | 0.964 | 0.962 | 14.560 |
| Isolation Forest | A: normal-only novelty | 0.961 | 0.951 | 1.877 |
| Self-training RF (2% labels) | B: few-label self-training | 0.953 | 0.948 | 91.487 |
| Autoencoder (PyTorch) | A: normal-only novelty | 0.951 | 0.958 | 19.426 |
| PCA reconstruction | A: normal-only novelty | 0.949 | 0.958 | 0.737 |
| One-Class SVM | A: normal-only novelty | 0.945 | 0.910 | 1.764 |

| model | dos | false_alarm_rate | probe | r2l | u2r |
|---|---|---|---|---|---|
| Supervised RF (known attacks) | 0.998 | 0.131 | 1.000 | 0.902 | 0.940 |
| Isolation Forest | 0.769 | 0.025 | 0.969 | 0.085 | 0.328 |
| Self-training RF (2% labels) | 0.989 | 0.142 | 0.999 | 0.858 | 0.657 |
| Autoencoder (PyTorch) | 0.848 | 0.042 | 0.779 | 0.347 | 0.687 |
| PCA reconstruction | 0.828 | 0.035 | 0.693 | 0.321 | 0.627 |
| One-Class SVM | 0.912 | 0.077 | 0.939 | 0.441 | 0.761 |


**Honest reading of these numbers**
. **The meaningful experiment is NSL-KDD**, where R2L and U2R are famously hard and `KDDTest+` contains attack types absent from `KDDTrain+`.
- What to look for on real data: the supervised model's detection rate on the novel/rare families vs. the novelty detectors, and how much the 2%-label model closes the gap.

## Layout
```
src/netad/  data.py · models.py · autoencoder.py · evaluate.py · train.py
tests/      schema, novel-family split, unseen categories, threshold/tie handling, end-to-end
```
MIT licensed.
