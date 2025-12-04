# Autoencoder Training Report  
### Ticket: FPN-8 — Phase 2: Model Training & Evaluation  
### Model: Unsupervised Autoencoder for Anomaly Detection

---

## 1. Overview

This report summarizes the training, evaluation, and behavior of the Autoencoder anomaly-detection model. The Autoencoder is trained **exclusively on legitimate (Class = 0) transactions**, learning the structure of normal behavior and identifying deviations as anomalies. Its purpose is to complement the supervised XGBoost classifier by catching *previously unseen or emerging fraud patterns* in a pre-authorization e-commerce environment.

---

## 2. Data Used

### **Training Input**
- Source: `data/processed/train.csv`
- Filter: Only legitimate transactions (`Class == 0`)
- Final training shape:  
  **X_train_legit = (13,971 × 7,621 features)**

### **Validation Input (for threshold selection)**
- Source: `data/processed/val.csv`
- Filter: `Class == 0`
- Shape: **X_val_legit = (2,995 × 7,621)**

### **Test Set**
- Source: `data/processed/test.csv`
- Full distribution preserved (fraud extremely rare)

The data has already undergone:
- enrichment  
- one-hot encoding  
- scaling  
- SMOTE (train only)  
- time-based splitting  

---

## 3. Model Architecture

A symmetric dense Autoencoder was built using TensorFlow/Keras:

```
Input (7621)
 ↓
Dense(64, relu)
Dense(32, relu)
Dense(16, relu)     ← Bottleneck layer
Dense(32, relu)
Dense(64, relu)
Output(7621, linear)
```

- **Loss:** Mean Squared Error (MSE)  
- **Optimizer:** Adam  
- **Purpose:** Reconstruct legitimate transactions with minimal error; anomalies → higher reconstruction error.

---

## 4. Training Configuration

| Parameter | Value |
|----------|--------|
| Epochs | 100 |
| Batch size | 512 |
| Validation split | 10% of legit training data |
| Early stopping | Patience = 5, monitor = val_loss |
| Random seed | Ensures reproducibility |

Training stabilized smoothly with a clean convergence curve.  
Early stopping triggered after **epoch ~15–20**, indicating good generalization.

A loss curve was saved as:

```
artifacts/plots/ae_loss_curve.png
```

---

## 5. Reconstruction Error Analysis

### Error statistics (legitimate validation set):

| Metric | Value |
|--------|--------|
| Mean | ~0.0031 |
| Std | ~0.0023 |
| 95th percentile | **0.0067** ← chosen threshold |

### Threshold Selection  
Threshold = **95th percentile** of reconstruction errors on legitimate validation samples.  
Meaning:

> Only the top 5% most abnormal legitimate transactions exceed this error, keeping the false positive rate controlled.

---

## 6. Evaluation on Validation and Test Data

### 📌 Key Definitions:
- **TPR / Recall**: ability to catch fraud  
- **FPR**: how often legit users are flagged incorrectly  
- **Precision**: how many flagged transactions are truly fraud  
- **ROC-AUC / PR-AUC**: ranking-quality metrics

---

## 6.1 Validation Set Results

| Metric | Value |
|--------|--------|
| Threshold | **0.0067** |
| Precision | 0.0260 |
| Recall (TPR) | **0.8000** |
| FPR | **0.050** |
| ROC-AUC | 0.9513 |
| PR-AUC | 0.1072 |

**Confusion Matrix**
```
[[2845   150]   TN FP
 [   1     4]]  FN TP
```

Interpretation:
- Caught **4/5 frauds** (1 missed → recall = 0.8)
- False positives ~5% → acceptable for anomaly detection  
- Precision low due to extremely rare fraud (expected)

---

## 6.2 Test Set Results

| Metric | Value |
|--------|--------|
| Threshold | **0.0067** |
| Precision | 0.0179 |
| Recall (TPR) | **1.0000** |
| FPR | **0.0367** |
| ROC-AUC | 0.9863 |
| PR-AUC | 0.0343 |

**Confusion Matrix**
```
[[2888   110]
 [   0     2]]
```

Interpretation:
- Perfect recall on test set (2/2 frauds caught)  
- Only 110 false positives out of 2998 legitimate transactions (~3.7%)  

---

## 7. Error Distribution Plot

A histogram comparing legitimate vs fraud reconstruction errors clearly shows **fraud outliers cluster to the right**, validating anomaly-detection behavior.

Saved at:

```
artifacts/plots/ae_error_distribution.png
```

---

## 8. Feature-Level Error Analysis

For fraud samples, the following features contributed the highest anomaly scores:

- Velocity features (`txn_count_5m`, `txn_count_30m`, `txn_count_60m`)
- Behavioral features (`account_age_days`, `token_age_days`)
- Risk indicators (`ip_reputation`, `is_proxy_vpn`)
- Geographic anomalies (`geo_distance_km`)
- Temporal features (`night_txn`, `weekend_txn`)

These align with real-world fraud indicators.

---

## 9. Final Artifacts

| File | Purpose |
|------|---------|
| `autoencoder_model.keras` | Final trained autoencoder |
| `ae_threshold.txt` | Threshold for anomaly classification |
| `ae_metrics.json` | Complete evaluation metrics |
| `plots/ae_loss_curve.png` | Training dynamics |
| `plots/ae_error_distribution.png` | Fraud vs legit separation |

---

## 10. Conclusion

The Autoencoder successfully learned the structure of legitimate transactions and acts as an effective anomaly detector. It:

- Achieves **high recall (0.80–1.00)**  
- Maintains **FPR well below 10%**  
- Distinguishes fraud patterns via reconstruction error  
- Provides a complementary unsupervised signal alongside supervised XGBoost  

This completes Ticket FPN-8 and prepares the system for model fusion in Phase 3.

