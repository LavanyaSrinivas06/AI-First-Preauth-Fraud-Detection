# Preprocessing Summary

## Dataset Sizes (after preprocessing)

- Train (before SMOTE): 199364 rows
- Train (after SMOTE): 397960 rows
- Validation: 42721 rows
- Test: 42722 rows

## Class Distribution

- Train before SMOTE: 0: 198980 (99.81%), 1: 384 (0.19%)
- Train after SMOTE: 0: 198980 (50.00%), 1: 198980 (50.00%)
- Validation: 0: 42665 (99.87%), 1: 56 (0.13%)
- Test: 0: 42670 (99.88%), 1: 52 (0.12%)

## Feature Groups

- Numerical features (39): V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14, V15, V16, V17, V18, V19, V20, V21, V22, V23, V24, V25, V26, V27, V28, Amount, ip_reputation, txn_count_5m, txn_count_30m, txn_count_60m, avg_amount_7d, account_age_days, token_age_days, avg_spend_user_30d, geo_distance_km, amount_zscore
- Categorical features (10): device_os, browser, is_new_device, ip_country, is_proxy_vpn, billing_country, shipping_country, country_mismatch, night_txn, weekend_txn
