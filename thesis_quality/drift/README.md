## Drift Monitoring (PSI)

- Baseline: training dataset
- Current: simulated production drift
- Method: Population Stability Index (PSI)
- Thresholds:
  - PSI < 0.1: stable
  - 0.1–0.25: moderate drift
  - > 0.25: significant drift

This module demonstrates the system’s ability to detect
feature distribution shifts prior to model retraining.
