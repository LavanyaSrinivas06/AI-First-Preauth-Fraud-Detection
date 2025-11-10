# Data Dictionary

## A. Base Fields (Kaggle Credit Card Fraud dataset)
- **Time** *(float/int)* — seconds since the first transaction
- **V1–V28** *(float)* — anonymized PCA components
- **Amount** *(float)* — transaction amount
- **Class** *(int: 0/1)* — fraud label (1 = fraud)

## B. Enrichment Fields (to be generated later)
**Device / Browser**
- `device_id` *(string)* — anonymized device identifier
- `device_os` *(string)* — OS family
- `browser` *(string)* — browser family
- `is_new_device` *(bool)* — first time device for account

**Network / IP**
- `ip_country` *(string, ISO-2)* — country inferred from IP
- `is_proxy_vpn` *(bool)* — proxy/VPN/Tor flag
- `ip_reputation` *(float 0–1)* — IP risk score

**Behavioral / Velocity**
- `txn_count_5m`, `txn_count_30m`, `txn_count_60m` *(int)* — recent txn counts
- `avg_amount_7d` *(float)* — 7-day rolling average amount

**Account / Token Age**
- `account_age_days`, `token_age_days` *(int)* — entity ages
- `avg_spend_user_30d` *(float)* — 30-day avg user spend

**Geo / Address**
- `billing_country`, `shipping_country` *(string, ISO-2)*
- `geo_distance_km` *(float)* — distance between billing vs shipping

**Derived**
- `amount_zscore` *(float)*
- `night_txn` *(bool)*
- `weekend_txn` *(bool)*
