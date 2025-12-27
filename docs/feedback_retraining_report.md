# Feedback Retraining Report

- Base train rows: 397960
- Feedback rows added: 1
- Combined rows used: 397961

## Validation
- ROC-AUC: 0.9985037082921766
- PR-AUC: 0.9082987891957146
- F1@0.5: 0.8990825688073395

## Test
- ROC-AUC: 0.9935876403886715
- PR-AUC: 0.8462771603653648
- F1@0.5: 0.8155339805825242

## Artifacts
- Model: `artifacts/xgb_model_feedback.pkl`
- Metrics: `artifacts/xgb_feedback_metrics.json`

Notes: This retraining run demonstrates the proposal-aligned feedback loop (human labels → dataset → model update).