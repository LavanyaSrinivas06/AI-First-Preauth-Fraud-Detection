# AI-First Preauth Fraud Detection for Secure E-Commerce Checkout

An AI-powered system that detects fraudulent behavior **before payment authorization** during an e-commerce checkout process.

---

## 🔍 Problem Statement

E-commerce platforms face increasing fraud risk during high-volume transactions. This project introduces a **Preauth Fraud Detector** that runs before payment is completed and prevents the transaction if fraud is detected.

---

## 🧠 Features

- ✅ Supervised & unsupervised fraud detection models
- ✅ Real-time prediction before payment proceeds
- ✅ Ticket generation system for flagged cases
- ✅ FastAPI backend for real-time inference
- ✅ Pattern-based fraud signals engineered from research

---

## 📁 Project Structure

AI-First-Preauth-Fraud-Detection/
│
├── data/
│ ├── raw/ # Synthetic/generated raw datasets
│ └── processed/ # Cleaned, feature-ready datasets
│
├── notebooks/ # Data exploration and training notebooks
│ ├── 01_data_exploration.ipynb
│ └── 02_model_training.ipynb
│
├── scripts/ # Automation and data scripts
│ └── generate_synthetic_fraud_data.py
│
├── models/ # Trained model files
│
├── api/ # FastAPI backend
│ ├── main.py
│ └── schema.py
│
├── tickets/ # Logs of blocked transactions
│ └── blocked_log.json
│
├── utils/ # Utility functions
│ └── model_utils.py
│
├── README.md
├── requirements.txt
├── .gitignore
└── setup.sh / setup.bat


---

## 🧪 Synthetic Dataset Generation

This project uses a synthetically generated dataset that simulates real-world e-commerce checkout data. The data includes key behavioral, transactional, and rule-based fraud patterns.

- **Script**: [`scripts/generate_synthetic_fraud_data.py`](scripts/generate_synthetic_fraud_data.py)
- **Outputs**:
  - `data/raw/ecom_synthetic_fraud_dataset_v1.csv`
  - `data/raw/ecom_synthetic_fraud_dataset_v1.json`

> ⚠️ Fraud patterns are derived from academic and industry references, such as:
> - *Khalil et al., 2021 - Pattern Analysis for Transaction Fraud Detection* [[Link](https://www.researchgate.net/publication/350149868)]

---

## 🛠️ Getting Started

1. Clone the repo  
2. Set up virtual environment  
3. Run the data generation script:
   ```bash
   python scripts/generate_synthetic_fraud_data.py
