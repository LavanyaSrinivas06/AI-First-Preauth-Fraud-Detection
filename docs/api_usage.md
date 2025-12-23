## Hybrid scoring (XGBoost + Autoencoder)

We compute:

P_final = ALPHA * P_xgb + (1 - ALPHA) * P_ae

- `P_xgb`: supervised fraud probability from XGBoost
- `P_ae`: anomaly risk derived from AE reconstruction error thresholds:
  - < review threshold -> 0.0
  - between review and block -> 0.5
  - >= block threshold -> 1.0

Decision:
- approve if P_final <= T_LOW
- block if P_final >= T_HIGH
- else review

Config via env:
- `ALPHA` (default 0.7)
- `T_LOW` (default 0.25)
- `T_HIGH` (default 0.75)
- `USE_AE_ONLY_IN_GRAYZONE` (default true)
