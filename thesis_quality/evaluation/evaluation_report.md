# Model Evaluation

## Evaluation Protocol
(Summarize protocol.md in 6–8 lines)

## Dataset
- Train / Validation / Test split
- Test set size
- Class imbalance note

## Models Evaluated
- XGBoost (supervised)
- Autoencoder (unsupervised)
- Hybrid usage note (AE as gray-zone gate)

## Metrics
- ROC-AUC
- PR-AUC (primary)
- Precision / Recall
- Thresholding strategy

## Quantitative Results
(Embed benchmark_table.csv as a table)

## ROC & PR Curves
(Reference plots)

## Key Observations
- XGB strengths
- AE behavior
- Why hybrid makes sense

## Operational Implications
- Fraud recall vs false positives
- Why PR is prioritized over ROC
