### Overall (test set incl. novel attacks)

| model | regime | pr_auc | roc_auc | fit_s |
|---|---|---|---|---|
| Supervised RF (known attacks) | C: fully supervised | 0.964 | 0.962 | 14.560 |
| Isolation Forest | A: normal-only novelty | 0.961 | 0.951 | 1.877 |
| Self-training RF (2% labels) | B: few-label self-training | 0.953 | 0.948 | 91.487 |
| Autoencoder (PyTorch) | A: normal-only novelty | 0.951 | 0.958 | 19.426 |
| PCA reconstruction | A: normal-only novelty | 0.949 | 0.958 | 0.737 |
| One-Class SVM | A: normal-only novelty | 0.945 | 0.910 | 1.764 |

### Detection rate per attack family at 5% false-alarm budget

| model | dos | false_alarm_rate | probe | r2l | u2r |
|---|---|---|---|---|---|
| Supervised RF (known attacks) | 0.998 | 0.131 | 1.000 | 0.902 | 0.940 |
| Isolation Forest | 0.769 | 0.025 | 0.969 | 0.085 | 0.328 |
| Self-training RF (2% labels) | 0.989 | 0.142 | 0.999 | 0.858 | 0.657 |
| Autoencoder (PyTorch) | 0.848 | 0.042 | 0.779 | 0.347 | 0.687 |
| PCA reconstruction | 0.828 | 0.035 | 0.693 | 0.321 | 0.627 |
| One-Class SVM | 0.912 | 0.077 | 0.939 | 0.441 | 0.761 |
