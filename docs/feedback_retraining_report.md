# Feedback Retraining Report

- DB used: `artifacts/stores/inference_store.sqlite`
- Base train rows: 397960
- Feedback rows added: 5
- Combined rows used: 397965

## Validation
- ROC-AUC: 0.9979612763891447
- PR-AUC: 0.9017066342218041
- F1@0.5: 0.8867924528301887

## Test
- ROC-AUC: 0.9947265237691767
- PR-AUC: 0.8408355155418017
- F1@0.5: 0.8155339805825242

## Artifacts
- Model: `artifacts/models/xgb_model_feedback.pkl`
- Metrics: `artifacts/metrics/xgb_feedback_metrics.json`

Notes: This run demonstrates the feedback loop: analyst labels → labeled dataset → updated model artifact.