# Model Comparison Report (XGBoost vs Autoencoder)

## Metrics Comparison Table
| Metric | XGBoost | Autoencoder |
|--------|---------|-------------|
| precision | 0.6667 | 0.0179 |
| recall | 1.0000 | 1.0000 |
| f1 | 0.8000 | 0.0351 |
| roc_auc | 1.0000 | 0.9863 |
| pr_auc | 1.0000 | 0.0343 |

## Confusion Matrices
**XGBoost:** [[2997, 1], [0, 2]]

**Autoencoder:** [[2888, 110], [0, 2]]

## Notes
- XGBoost excels at precision and stable detection of known fraud patterns.
- Autoencoder excels at recall (finding unseen patterns) but has more false positives.
- Ensemble fusion may be explored in Phase 3.
