# Data Card — Enriched Pre-Authorization Fraud Dataset
_Last updated: 2025-11-13_

## 1. Dataset Overview

**Name:** Enriched Pre-Authorization Fraud Detection Dataset  
**Purpose:** Used to train machine learning models for real-time fraud detection during the pre-authorization stage of e-commerce checkout.  
**Source:**  
- Base dataset derived from the well-known Credit Card Fraud dataset (Kaggle / European card transactions).  
- Enriched with synthetic contextual, behavioral, and risk features created in Ticket FPN-3 and FPN-4.

**Total Rows:** 284,807  
**Total Columns:** 51 (including PCA features, behavioral features, contextual features, and the target label)

---

## 2. Intended Use

This dataset is designed for:

- Training supervised ML models (e.g., XGBoost)  
- Training unsupervised anomaly detectors (e.g., Autoencoder)  
- Pre-authorization fraud scoring during online checkout  
- Research in fraud detection, feature engineering, and risk modeling  

It is **not** designed for:

- Post-authorization fraud modeling  
- Chargeback prediction  
- Identification or profiling of individuals  

---

## 3. Ethical Considerations

### **Privacy & GDPR**
- No personal information is included.  
- Device IDs, IPs, browser strings, and risk attributes are **synthetic** and anonymized.  
- No reversible identifiers are present.  
- Computation of features does not rely on PII.

### **Model Fairness & Bias**
Potential risks:
- Certain countries may appear riskier due to enrichment logic.
- Behavioral features (txn_count_Xm, avg_spend_user_30d) may introduce bias toward heavy users.
- Fraud patterns may be unbalanced or synthetic.

Mitigation:
- Use performance metrics broken down by segments later (Phase 4).
- Avoid rule-based assumptions embedded in features.

---

## 4. Dataset Composition

### **4.1 Columns Included**
**PCA Components (30):**  
Time, V1–V28, Amount

**Fraud Label:**  
Class ∈ {0, 1}

**Behavioral Features:**
- txn_count_5m  
- txn_count_30m  
- txn_count_60m  
- avg_amount_7d  
- avg_spend_user_30d  
- amount_zscore  

**Device Features:**
- device_id  
- device_os  
- browser  
- is_new_device  

**Network / IP Features:**
- ip_country  
- is_proxy_vpn  
- ip_reputation  

**Account Features:**
- account_age_days  
- token_age_days  

**Geographical Features:**
- billing_country  
- shipping_country  
- geo_distance_km  

**Temporal Flags:**
- night_txn  
- weekend_txn  

---

## 5. Statistical Summary (High-Level)

- **Class imbalance:**  
  - Fraud (1): Extremely minority  
  - Legit (0): Majority  
  - SMOTE applied **only on training split**, not in this enriched dataset.

- **Missingness:**  
  - 0% missing values across all 51 columns  
  - Verified in `data_validation_report.md`

- **Numeric feature ranges:**  
  - Amount ≥ 0  
  - ip_reputation ∈ [0, 1]  
  - account_age_days ≥ 0  
  - token_age_days ≥ 0  
  - geo_distance_km ≥ 0  

---

## 6. Data Quality & Validation Summary

Based on `validate_data.py`:

- ✔ All schema columns present  
- ✔ All dtypes match expectations  
- ✔ All numeric range checks passed  
- ✔ Class column contains only 0 and 1  
- ✔ No missing values  
- ✔ Dataset shape consistent with expectations (284,807 × 51)

---

## 7. Limitations

- PCA components are not interpretable features.  
- Behavioral features are synthetic approximations of typical fraud patterns (not real user data).  
- Dataset lacks real-world noise such as missing metadata, corrupted fields, or inconsistent timestamps.  
- Does not include cross-device history or customer lifetime behavior.

---

## 8. Versioning & Updates

| Version | Date | Notes |
|--------|------|-------|
| v1.0 | 2025-11-13 | First validated, enriched dataset created with Ticket FPN-3 → FPN-6 pipeline. |

---

## 9. Contact

Maintainer: Lavanya Srinivas  
Thesis Project: *AI-First Pre-Authorization Fraud Detection for Secure E-Commerce Checkout*  
Institution: SRH Hochschule Berlin (M.Sc. AI & Big Data)

