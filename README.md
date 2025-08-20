# AI-First Preauth Fraud Detection for Secure E-Commerce Checkout

An AI system to detect fraudulent behavior **before payment authorization** during an online checkout.  
The pipeline combines **supervised learning** (known fraud patterns) and **unsupervised anomaly detection** (unknown/emerging fraud), with a decision layer for Approve / Block / Manual Review.

---

## Table of Contents
- [Overview](#overview)
- [Planned Approach](#planned-approach)
- [Repository Structure](#repository-structure)
- [Environment Setup](#environment-setup)
- [Dataset Setup](#dataset-setup)
- [Phase 1: Data Exploration](#phase-1-data-exploration)
- [Next Phases](#next-phases)
- [How to Work (GitHub Flow)](#how-to-work-github-flow)
- [License and Citation](#license-and-citation)

---

## Overview
- **Goal:** Reduce fraud losses and friction by making risk decisions **before** payment authorization.
- **Key ideas:**  
  - Supervised model (XGBoost) learns known fraud patterns.  
  - Unsupervised model (Autoencoder) flags unusual activity.  
  - Decision layer aggregates scores and returns **Approve / Block / Manual Review** with reason codes.  
  - Feedback loop: anomalies reviewed and added back to training to improve over time.

---

## Planned Approach
1. **Data Acquisition** — Start with a public, labeled dataset (Kaggle Credit Card Fraud).  
2. **Data Enrichment** — Add realistic preauth features (device, cart, velocity) where missing.  
3. **Supervised Model** — Train XGBoost to output fraud probability.  
4. **Real-Time Detection Flow** — If probability is high/low, decide; if uncertain, route to unsupervised layer.  
5. **Unsupervised Model** — Train Autoencoder on normal transactions to detect anomalies.  
6. **Learning Loop** — Repeated anomaly patterns are labeled and used for retraining the supervised model.  
7. **Decision Layer** — Aggregate outputs and produce one of three outcomes with reason codes.  
8. **Evaluation** — Use PR-AUC, Recall at low FPR, ROC-AUC; monitor FP/FN rates.

> This design helps catch both **known fraud** and **new fraud types** that change over time.

---

## Repository Structure

AI-First-Preauth-Fraud-Detection/
├─ data/
│ ├─ raw/ # original datasets (not committed)
│ └─ processed/ # feature-ready / intermediate outputs
├─ notebooks/
│ ├─ eda_baseline.ipynb
│ └─ (more to come)
├─ docs/
│ ├─ images/ # saved figures for README/thesis
│ └─ diagrams/ # pipeline/flow diagrams
├─ src/
│ ├─ features/ # enrichment (device, cart, velocity)
│ ├─ train/ # training scripts
│ ├─ serve/ # routing + decision layer
│ └─ eval/ # evaluation & stress tests
├─ models/ # saved model artifacts (local)
├─ config/ # thresholds, settings
├─ requirements.txt
├─ README.md
└─ .gitignore


---

## Getting Started

e repository  
bash
   git clone https://github.com/LavanyaSrinivas06/AI-First-Preauth-Fraud-Detection.git
   cd AI-First-Preauth-Fraud-Detection

# install dependencies
pip install -r requirements.txt

# Dataset Setup

This project uses the **Credit Card Fraud Detection Dataset** from Kaggle:
[https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Format: ~284,000 transactions with anonymized PCA features (V1–V28), Amount, Time, and Class (fraud or not).
Synthetic Data: Additional enriched attributes may be generated to simulate real-world e-commerce signals.

Phase 1: Data Exploration

✔ Loaded Kaggle dataset
✔ Checked class imbalance (~0.2% fraud)
✔ Analyzed transaction amount and time patterns
✔ Correlation heatmap generated
✔ PCA v<img width="745" height="768" alt="fraud_detection_pipeline_vertical" src="https://github.com/user-attachments/assets/9be67c16-87ad-4414-9410-f0d9bd06884f" />
s t-SNE visualization done






![Uploading fraud_detection_pipeline_vertical.png…]()





