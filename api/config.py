# api/config.py
from __future__ import annotations

import os

APP_VERSION = "0.1.0"

# Hybrid fusion
ALPHA = float(os.getenv("FPN_ALPHA", "0.7"))

# Decision thresholds
T_LOW = float(os.getenv("FPN_T_LOW", "0.25"))
T_HIGH = float(os.getenv("FPN_T_HIGH", "0.75"))

# Paths
PREPROCESS_PATH = os.getenv("FPN_PREPROCESS_PATH", "artifacts/preprocess.joblib")
FEATURES_PATH = os.getenv("FPN_FEATURES_PATH", "artifacts/features.json")
XGB_PATH = os.getenv("FPN_XGB_PATH", "artifacts/xgb_model.pkl")
AE_PATH = os.getenv("FPN_AE_PATH", "artifacts/autoencoder_model.h5")
AE_THRESHOLD_PATH = os.getenv("FPN_AE_THRESHOLD_PATH", "artifacts/ae_threshold.txt")

# Logging
LOG_DIR = os.getenv("FPN_LOG_DIR", "logs")
INFERENCE_LOG_PATH = os.getenv("FPN_INFERENCE_LOG_PATH", f"{LOG_DIR}/inference.log")
