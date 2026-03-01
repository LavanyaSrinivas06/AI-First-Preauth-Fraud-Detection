# AI-First Pre-Authorization Fraud Detection

> **A hybrid XGBoost + Autoencoder system for real-time pre-authorization fraud detection with human-in-the-loop review and feedback-driven retraining.**

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.121-009688.svg)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.x-orange.svg)](https://xgboost.readthedocs.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00.svg)](https://www.tensorflow.org/)

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Key Results](#key-results)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Data](#data)
- [Models](#models)
- [API Reference](#api-reference)
- [Dashboard](#dashboard)
- [Evaluation & Statistical Validation](#evaluation--statistical-validation)
- [Notebooks](#notebooks)
- [Testing](#testing)
- [Reproducibility](#reproducibility)
- [Documentation](#documentation)
- [License](#license)

---

## Overview

This project implements an **AI-first fraud detection system** designed for the pre-authorization phase of e-commerce payment transactions. Rather than relying on a single classifier, the system employs a **hybrid architecture** that combines:

1. **XGBoost** — a supervised gradient-boosted tree model trained on labeled fraud/legitimate transactions for fast, high-precision classification.
2. **Autoencoder** — an unsupervised deep learning model that detects anomalous transactions in the *gray zone* (uncertain XGBoost predictions) via reconstruction error analysis.
3. **Human-in-the-Loop Review** — uncertain cases are routed to a review queue for manual analyst decisions.
4. **Feedback Loop** — analyst decisions are captured and used to periodically retrain the XGBoost model, creating a continuously improving system.

The system is served as a **FastAPI REST API** and visualized through a **Streamlit dashboard**.

---

## Architecture

<img width="4102" height="2190" alt="image" src="https://github.com/user-attachments/assets/713fdfd9-5da2-4d9a-abcb-8d007e0f4c19" />


### Decision Flow

```text
E-commerce Checkout
        │
        ▼
  POST /preauth   ──►  Fraud Detection API
        │
        ▼
   XGBoost Model
   P(fraud) score
        │
        ├── P < 0.05 (Low Risk)  ──────────►  ✅ APPROVE  ──►  Payment Gateway
        │
        ├── P > 0.80 (High Risk) ──────────►  🚫 BLOCK   ──►  Decline (HTTP 500)
        │
        └── 0.05 ≤ P ≤ 0.80 (Gray Zone)
                    │
                    ▼
              Autoencoder
           Reconstruction Error
                    │
                    ├── Error ≥ 4.896  ────►  🚫 BLOCK
                    ├── Error ≥ 0.692  ────►  👁️ REVIEW  ──►  Human-in-the-Loop
                    └── Error < 0.692  ────►  ✅ APPROVE
                                                    │
                                                    ▼
                                              Feedback Store
                                                    │
                                                    ▼
                                            Retrain XGBoost
                                                    │
                                                    ▼
                                             Model Registry
```

### Hybrid Scoring Logic

For transactions in the gray zone (`USE_AE_ONLY_IN_GRAYZONE=true`):

$$P_{\text{final}} = \alpha \cdot P_{\text{xgb}} + (1 - \alpha) \cdot P_{\text{ae}}$$

Where:

- $P_{\text{xgb}}$: supervised fraud probability from XGBoost
- $P_{\text{ae}}$: anomaly risk from Autoencoder reconstruction error
- $\alpha = 0.7$ (default)

Autoencoder risk mapping:
| Reconstruction Error | $P_{\text{ae}}$ |
|---|---|
| < 0.692 (review threshold) | 0.0 |
| 0.692 – 4.896 | 0.5 |
| ≥ 4.896 (block threshold) | 1.0 |

---

## Key Results

### XGBoost Performance (Test Set — 42,722 transactions)

| Metric | Value |
|---|---|
| **ROC-AUC** | 0.9911 |
| **Precision** | 0.875 |
| **Recall** | 0.8077 |
| **F1-Score** | 0.840 |
| **Optimal Threshold** | 0.307 |

### Confusion Matrix (Test)

|  | Predicted Legit | Predicted Fraud |
|---|---|---|
| **Actual Legit** (42,670) | 42,664 | 6 |
| **Actual Fraud** (52) | 10 | 42 |

### Autoencoder Thresholds

| Threshold | Percentile | Reconstruction Error |
|---|---|---|
| Review | 95th | 0.6916 |
| Block | 99.5th | 4.8956 |

### Production Decision Thresholds

| Parameter | Value | Description |
|---|---|---|
| `xgb_threshold` | 0.307 | Optimal classification threshold |
| `xgb_t_low` | 0.05 | Below → auto-approve |
| `xgb_t_high` | 0.80 | Above → auto-block |
| `ae_review` | 0.692 | AE error above → send to review |
| `ae_block` | 4.896 | AE error above → auto-block |

### Feedback-Retrained Model

| Metric | Base Model | After Feedback |
|---|---|---|
| ROC-AUC (Test) | 0.9911 | 0.9941 |
| F1 @ 0.5 (Test) | 0.840 | 0.812 |

---

## Project Structure

```
.
├── api/                          # FastAPI REST API
│   ├── main.py                   # Application factory
│   ├── core/                     # Config, logging, error handlers
│   ├── routers/                  # Endpoint handlers
│   │   ├── health.py             #   GET  /health
│   │   ├── preauth.py            #   POST /preauth
│   │   ├── review.py             #   GET/POST /review
│   │   └── feedback.py           #   POST /feedback
│   ├── schemas/                  # Pydantic request/response models
│   └── services/                 # Model service, reason builder, store
│
├── dashboard/                    # Streamlit monitoring dashboard
│   ├── app.py                    # Main dashboard entry point
│   ├── api.py                    # API client for dashboard
│   └── pages/                    # Multi-page dashboard views
│       ├── home.py               #   Overview metrics
│       ├── model_overview.py     #   Model performance
│       ├── queue.py              #   Review queue
│       └── review.py             #   Transaction review detail
│
├── src/                          # Core ML source code
│   ├── models/                   # Model registry
│   ├── preprocess/               # Preprocessing pipeline (ColumnTransformer)
│   └── utils/                    # Shared utilities
│
├── scripts/                      # Training, evaluation & ops scripts
│   ├── train_xgboost.py          # XGBoost training with Optuna tuning
│   ├── train_autoencoder.py      # Autoencoder training
│   ├── run_preprocessing.py      # Data preprocessing pipeline
│   ├── compare_models.py         # Model comparison
│   ├── ablation_study.py         # Ablation: XGB vs Threshold vs Hybrid
│   ├── statistical_significance_test.py  # McNemar's test + bootstrap CIs
│   ├── explainability_shap.py    # SHAP feature explanations
│   ├── feedback_loop.py          # Feedback ingestion
│   ├── retrain_from_feedback.py  # Feedback-driven retraining
│   ├── reproduce.py              # Full reproducibility script
│   ├── validate_data.py          # Data quality checks
│   ├── drift/                    # Data drift detection
│   ├── eval/                     # Evaluation utilities
│   ├── ops/                      # Operational scripts
│   └── perf/                     # Performance benchmarks
│
├── notebooks/                    # Jupyter notebooks (thesis narrative)
│   └── 00_thesis_story/
│       ├── 00_data_overview.ipynb
│       ├── 01_preprocessing_pipeline.ipynb
│       ├── 02_enrichment_simulation.ipynb
│       ├── 03_feature_engineering_summary.ipynb
│       ├── 04_xgboost_training_and_eval.ipynb
│       ├── 05_autoencoder_training_and_eval.ipynb
│       ├── 06_model_comparison.ipynb
│       └── 07_explainability_shap.ipynb
│
├── artifacts/                    # Trained models, metrics & outputs
│   ├── models/                   # Serialized models
│   │   ├── xgb_model.pkl         #   XGBoost (primary)
│   │   ├── xgb_model_feedback.pkl #  XGBoost (feedback-retrained)
│   │   └── autoencoder_model.keras # Autoencoder
│   ├── preprocess/               # Fitted preprocessor + feature config
│   ├── thresholds/               # AE threshold calibration
│   ├── metrics/                  # Evaluation metrics
│   ├── plots/                    # Generated plots
│   ├── explainability/           # SHAP values & explanations
│   ├── reports/                  # Generated reports
│   └── stores/                   # Feedback & review stores (SQLite)
│
├── data/
│   ├── raw/                      # Original dataset
│   └── processed/                # Train/val/test splits (preprocessed)
│
├── tests/                        # Test suite
│   ├── unit/                     # Unit tests (model service, config, store)
│   ├── integration/              # API integration tests
│   ├── conftest.py               # Shared fixtures
│   └── test_smoke.py             # Smoke tests
│
├── docs/                         # Documentation
│   ├── figures/                  # Architecture & thesis diagrams
│   ├── robustness/               # Robustness analysis docs
│   ├── data_card.md              # Dataset documentation
│   ├── data_dictionary.md        # Feature definitions
│   ├── preprocessing_summary.md  # Preprocessing pipeline details
│   ├── model_comparison_report.md # XGBoost vs Hybrid analysis
│   ├── feedback_loop.md          # Feedback loop design
│   └── api_usage.md              # API usage guide
│
├── requirements.txt              # Python dependencies
├── requirements.lock.txt         # Locked dependency versions
├── settings.yaml                 # Configuration
├── pytest.ini                    # Test configuration
├── sample_request.json           # Example API payload
├── REPRODUCE.md                  # Reproducibility instructions
└── LICENSE                       # MIT License
```

---

## Setup & Installation

### Prerequisites

- **Python 3.10+** (tested on 3.10)
- **macOS** (Apple Silicon) or Linux
- ~2 GB disk space for models and data

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/AI-First-Preauth-Fraud-Detection.git
cd AI-First-Preauth-Fraud-Detection

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# For Apple Silicon Macs (M1/M2/M3):
pip install tensorflow-macos tensorflow-metal
```

> **Note:** For exact reproducibility, use `requirements.lock.txt` instead of `requirements.txt`.

### Key Dependencies

| Package | Purpose |
|---|---|
| `xgboost` | Supervised fraud classifier |
| `tensorflow` / `keras` | Autoencoder anomaly detector |
| `scikit-learn` | Preprocessing (ColumnTransformer) |
| `fastapi` + `uvicorn` | REST API server |
| `streamlit` | Monitoring dashboard |
| `shap` | Model explainability |
| `statsmodels` | Statistical significance testing |
| `optuna` | Hyperparameter optimization |
| `wandb` | Experiment tracking |
| `pytest` | Testing framework |

---

## Data

The project uses a synthetic fraud detection dataset derived from realistic transaction patterns.

- **Total transactions:** ~213,000 (after enrichment)
- **Fraud rate:** ~0.12% (highly imbalanced)
- **Features:** 28 PCA components (V1–V28) + Amount + 22 enriched features
- **Splits:** Train / Validation / Test (pre-split in `data/processed/`)

### Feature Categories

| Category | Features | Description |
|---|---|---|
| PCA Components | V1 – V28 | Anonymized transaction features |
| Transaction | Amount, Amount Z-score | Transaction value and deviation |
| Velocity | txn_count_5m, _30m, _60m | Transaction frequency windows |
| Device/Network | device_os, browser, is_new_device, ip_reputation, is_proxy_vpn | Client fingerprinting |
| Geographic | ip_country, billing_country, shipping_country, geo_distance_km, country_mismatch | Location analysis |
| Account | account_age_days, token_age_days, avg_spend_user_30d, avg_amount_7d | Account history |
| Temporal | night_txn, weekend_txn | Time-of-day signals |

See [`docs/data_dictionary.md`](docs/data_dictionary.md) and [`docs/data_card.md`](docs/data_card.md) for full details.

---

## Models

### XGBoost (Primary Classifier)

- **Type:** Supervised gradient-boosted tree
- **Tuning:** Optuna Bayesian hyperparameter search
- **Best Hyperparameters:**
  - `max_depth`: 6
  - `n_estimators`: 250
  - `learning_rate`: 0.05
  - `subsample`: 0.8
  - `colsample_bytree`: 1.0
- **Artifact:** `artifacts/models/xgb_model.pkl`

### Autoencoder (Gray-Zone Anomaly Detector)

- **Type:** Undercomplete autoencoder (trained on legitimate transactions only)
- **Purpose:** Detect anomalous patterns in the XGBoost "uncertain" zone (0.05 ≤ P ≤ 0.80)
- **Threshold Calibration:**
  - Review: 95th percentile of validation reconstruction errors → **0.692**
  - Block: 99.5th percentile → **4.896**
- **Artifact:** `artifacts/models/autoencoder_model.keras`

### Preprocessing Pipeline

- **Type:** scikit-learn `ColumnTransformer`
  - Numeric features → `StandardScaler`
  - Categorical features → `OneHotEncoder`
- **Artifact:** `artifacts/preprocess/preprocess.joblib`

---

## API Reference

Start the API server:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check — returns model status |
| `POST` | `/preauth` | Score a transaction — returns decision + risk |
| `GET` | `/review` | List transactions pending review |
| `POST` | `/review/{txn_id}` | Submit analyst decision for a review case |
| `POST` | `/feedback` | Submit feedback for model retraining |

### Example Request

```bash
curl -X POST http://localhost:8000/preauth \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

**Sample payload** (`sample_request.json`):

```json
{
  "V1": -1.504, "V2": -1.047, "V3": 2.659, ..., "V28": -0.127,
  "Amount": 118.58,
  "ip_reputation": 0.289,
  "txn_count_5m": 1,
  "account_age_days": 638,
  "device_os": "iOS",
  "browser": "Chrome",
  "is_new_device": false,
  "country_mismatch": true,
  ...
}
```

**Response:**

```json
{
  "decision": "approve",
  "risk_score": 0.023,
  "reasons": ["Low fraud probability", "Known device"],
  "model_version": "xgb_v1"
}
```

---

## Dashboard

The Streamlit dashboard provides real-time monitoring of the fraud detection system.

```bash
streamlit run dashboard/app.py
```

### Dashboard Pages

| Page | Description |
|---|---|
| **Home** | Overview metrics, decision distribution, recent alerts |
| **Model Overview** | Model performance metrics, feature importance |
| **Review Queue** | Pending review cases for analyst triage |
| **Review Detail** | Detailed transaction view with explainability |

Access at: [http://localhost:8501](http://localhost:8501)

---

## Evaluation & Statistical Validation

### Model Comparison

```bash
python scripts/compare_models.py
```

Compares XGBoost standalone vs. Hybrid (XGBoost + Autoencoder) on precision, recall, F1, and ROC-AUC.

### Statistical Significance (McNemar's Test)

```bash
python scripts/statistical_significance_test.py
```

Performs:
- **McNemar's exact test** — tests whether the disagreement between XGBoost and Hybrid predictions is statistically significant
- **Bootstrap confidence intervals** — 95% CIs for ΔPrecision, ΔRecall, and ΔF1

Input: `artifacts/test_predictions.csv` (42,723 per-transaction predictions)

### Ablation Study

```bash
python scripts/ablation_study.py
```

Three-configuration ablation:
1. **XGBoost Baseline** — single threshold at 0.307
2. **Threshold-Only** — three-tier thresholds (0.05 / 0.80) without Autoencoder
3. **Full Hybrid** — XGBoost + Autoencoder gray-zone routing

Pairwise McNemar's tests quantify each component's contribution.

### SHAP Explainability

```bash
python scripts/explainability_shap.py
```

Generates SHAP summary plots, feature importance rankings, and per-transaction explanations.

---

## Notebooks

The `notebooks/00_thesis_story/` directory contains a sequential narrative of the entire project:

| # | Notebook | Topic |
|---|---|---|
| 00 | `00_data_overview.ipynb` | Dataset exploration & class distribution |
| 01 | `01_preprocessing_pipeline.ipynb` | Feature engineering & preprocessing |
| 02 | `02_enrichment_simulation.ipynb` | Synthetic feature enrichment |
| 03 | `03_feature_engineering_summary.ipynb` | Feature summary & selection |
| 04 | `04_xgboost_training_and_eval.ipynb` | XGBoost training, tuning & evaluation |
| 05 | `05_autoencoder_training_and_eval.ipynb` | Autoencoder training & threshold calibration |
| 06 | `06_model_comparison.ipynb` | XGBoost vs Hybrid comparison |
| 07 | `07_explainability_shap.ipynb` | SHAP-based model interpretability |

---

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=api --cov=src --cov-report=html

# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/
```

### Test Structure

| Directory | Tests | Coverage |
|---|---|---|
| `tests/unit/` | Model service, config, store, error handling | Core ML logic |
| `tests/integration/` | API endpoints (health, preauth, review, feedback) | End-to-end API |
| `tests/test_smoke.py` | Import & basic sanity checks | Smoke tests |

---

## Reproducibility

See [`REPRODUCE.md`](REPRODUCE.md) for full instructions. Quick start:

```bash
# 1. Environment setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock.txt

# 2. Preprocessing
python scripts/run_preprocessing.py

# 3. Train models
python scripts/train_xgboost.py
python scripts/train_autoencoder.py

# 4. Evaluate
python scripts/compare_models.py

# 5. Run API
uvicorn api.main:app --port 8000

# 6. Run automated reproducibility check
python scripts/reproduce.py --setup-env --install --dry-run
```

---

## Documentation

| Document | Description |
|---|---|
| [`docs/data_card.md`](docs/data_card.md) | Dataset provenance & statistics |
| [`docs/data_dictionary.md`](docs/data_dictionary.md) | Feature definitions |
| [`docs/preprocessing_summary.md`](docs/preprocessing_summary.md) | Preprocessing pipeline details |
| [`docs/model_comparison_report.md`](docs/model_comparison_report.md) | XGBoost vs Hybrid analysis |
| [`docs/autoencoder_training_report.md`](docs/autoencoder_training_report.md) | AE training & threshold calibration |
| [`docs/feedback_loop.md`](docs/feedback_loop.md) | Feedback loop design & retraining |
| [`docs/feedback_retraining_report.md`](docs/feedback_retraining_report.md) | Post-retraining metrics |
| [`docs/api_usage.md`](docs/api_usage.md) | API hybrid scoring logic |
| [`docs/enrichment_report.md`](docs/enrichment_report.md) | Feature enrichment methodology |
| [`docs/data_validation_report.md`](docs/data_validation_report.md) | Data quality validation results |

---

## License

This project is licensed under the **MIT License** — see [`LICENSE`](LICENSE) for details.

© 2026 Lavanya Srinivas
