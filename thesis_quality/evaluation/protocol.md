# Evaluation Protocol

## Goal
Evaluate the fraud preauthorization decision engine models on a fixed holdout set and report standard classification metrics and plots for thesis reproducibility.

## Data splits (holdout policy)
- Data source: synthetic preauthorization dataset.
- Splits: `train.csv`, `val.csv`, `test.csv` under `data/processed/`.
- Policy:
  - **Train**: used for model fitting (XGBoost supervised; Autoencoder trained on legitimate-only subset).
  - **Validation**: used for threshold selection / operating point tuning.
  - **Test (holdout)**: used only for final reporting.

## Models evaluated
1. **XGBoost (supervised classifier)**
   - Output: probability `p_xgb` for fraud class.
   - Threshold chosen on validation set to optimize F1 (stored in `model_comparison_summary.json`).

2. **Autoencoder (unsupervised anomaly detector)**
   - Output: reconstruction error.
   - Threshold chosen using validation error distribution (stored in `model_comparison_summary.json`).

## Metrics reported
For both models on validation and test splits:
- Confusion matrix (TN, FP, FN, TP)
- Precision, Recall, F1-score
- ROC-AUC
- PR-AUC (Average Precision)

## Plots produced
Saved under `thesis_quality/evaluation/plots/`:
- ROC curves (XGB + comparison)
- PR curves (XGB + comparison)

## Reproducibility
- Inputs:
  - `data/processed/test.csv`
  - `artifacts/preprocess/features.json`
  - `artifacts/preprocess/preprocess.joblib`
  - `artifacts/models/xgb_model.pkl`
  - `artifacts/ae_errors/ae_test_errors.npy`
  - `artifacts/thresholds/ae_threshold.txt`
- Script:
  - `thesis_quality/evaluation/run_evaluation.py` (collects outputs into thesis folder)

## Output artifacts
- Metrics:
  - `thesis_quality/evaluation/metrics/model_comparison_summary.json`
  - `thesis_quality/evaluation/benchmark_table.csv`
- Plots:
  - `thesis_quality/evaluation/plots/*.png`
