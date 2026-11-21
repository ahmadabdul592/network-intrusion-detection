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
> From the **synthetic** data (scikit-learn autoencoder). **Replace with your NSL-KDD results** and add `results/per_family_detection.png`.

Detection rate per family at a ~5% false-alarm budget (R2L is unseen in training):

| Model | DoS | Probe | R2L (novel) | False alarms |
|---|---|---|---|---|
| Self-training RF (2% labels) | 1.00 | 1.00 | 1.00 | 0.004 |
| Supervised RF | 1.00 | 1.00 | 1.00 | 0.003 |
| One-Class SVM | 1.00 | 1.00 | 1.00 | 0.053 |
| Autoencoder (sklearn) | 1.00 | 1.00 | 0.67 | 0.049 |
| Isolation Forest | 1.00 | 1.00 | 0.40 | 0.056 |
| PCA reconstruction | 0.54 | 0.69 | 0.33 | 0.053 |

**Honest reading of these numbers**
- The synthetic R2L family is easy for tree models to separate, because its features fall outside the normal range. So this table mostly validates the *pipeline*. **The meaningful experiment is NSL-KDD**, where R2L and U2R are famously hard and `KDDTest+` contains attack types absent from `KDDTrain+`.
- What to look for on real data: the supervised model's detection rate on the novel/rare families vs. the novelty detectors, and how much the 2%-label model closes the gap.

## Limitations & next steps
- Threshold calibration on normal-only validation data assumes the validation traffic is representative.
- Add per-attack-type breakdown beyond the 4 families, SHAP for flagged flows, and a CICIDS2017 benchmark.
- Evaluate concept drift: train on older traffic, test on newer.

## Layout
```
src/netad/  data.py · models.py · autoencoder.py · evaluate.py · train.py
tests/      schema, novel-family split, unseen categories, threshold/tie handling, end-to-end
```
MIT licensed.
