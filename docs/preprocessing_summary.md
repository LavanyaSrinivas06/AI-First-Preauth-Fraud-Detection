# Preprocessing Summary

## Dataset Sizes (after preprocessing)

- Train (before SMOTE): 14000 rows
- Train (after SMOTE): 27942 rows
- Validation: 3000 rows
- Test: 3000 rows

## Class Distribution

- Train before SMOTE: 0: 13971 (99.79%), 1: 29 (0.21%)
- Train after SMOTE: 0: 13971 (50.00%), 1: 13971 (50.00%)
- Validation: 0: 2995 (99.83%), 1: 5 (0.17%)
- Test: 0: 2998 (99.93%), 1: 2 (0.07%)

## Feature Groups

- Numerical features (39): V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14, V15, V16, V17, V18, V19, V20, V21, V22, V23, V24, V25, V26, V27, V28, Amount, ip_reputation, txn_count_5m, txn_count_30m, txn_count_60m, avg_amount_7d, account_age_days, token_age_days, avg_spend_user_30d, geo_distance_km, amount_zscore
- Categorical features (10): device_id, device_os, browser, is_new_device, ip_country, is_proxy_vpn, billing_country, shipping_country, night_txn, weekend_txn
