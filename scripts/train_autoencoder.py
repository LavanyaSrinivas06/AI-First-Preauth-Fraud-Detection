#!/usr/bin/env python3
"""
Train an unsupervised Autoencoder on legitimate (non-fraud) transactions.

Uses NO-SMOTE processed splits (IMPORTANT):
- data/processed/train_nosmote.csv
- data/processed/val_nosmote.csv
- data/processed/test_nosmote.csv

Outputs:
- artifacts/autoencoder_model.keras
- artifacts/ae_thresholds.json          (two thresholds: review, block)
- artifacts/ae_metrics.json
- artifacts/plots/ae_loss_curve.png
- artifacts/plots/ae_error_distribution.png
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
ARTIFACT_DIR = ROOT / "artifacts"
PLOT_DIR = ARTIFACT_DIR / "plots"

ARTIFACT_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)

TRAIN_PATH = DATA_DIR / "train_nosmote.csv"
VAL_PATH   = DATA_DIR / "val_nosmote.csv"
TEST_PATH  = DATA_DIR / "test_nosmote.csv"

AE_MODEL_PATH = ARTIFACT_DIR / "autoencoder_model.keras"
AE_THRESHOLDS_PATH = ARTIFACT_DIR / "ae_thresholds.json"
AE_METRICS_PATH = ARTIFACT_DIR / "ae_metrics.json"

AE_LOSS_PLOT_PATH = PLOT_DIR / "ae_loss_curve.png"
AE_ERROR_DIST_PLOT_PATH = PLOT_DIR / "ae_error_distribution.png"


# -------------------------------------------------------------------
# Reproducibility
# -------------------------------------------------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def load_datasets():
    for p in [TRAIN_PATH, VAL_PATH, TEST_PATH]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train = train_df.drop(columns=["Class"])
    y_train = train_df["Class"]

    X_val = val_df.drop(columns=["Class"])
    y_val = val_df["Class"]

    X_test = test_df.drop(columns=["Class"])
    y_test = test_df["Class"]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def build_autoencoder(input_dim: int) -> models.Model:
    """
    Denoising AE + regularization to reduce overfitting / instability.
    Uses Huber loss (more stable than pure MSE for outliers).
    """
    inp = layers.Input(shape=(input_dim,))

    x = layers.GaussianNoise(0.05)(inp)

    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(64)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Dense(32, activation="relu")(x)  # bottleneck

    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)

    out = layers.Dense(input_dim, activation="linear")(x)

    model = models.Model(inp, out)
    opt = tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)
    model.compile(optimizer=opt, loss=tf.keras.losses.Huber(delta=1.0))
    return model


def reconstruction_errors(model: models.Model, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X).astype("float32")
    X_pred = model.predict(X, verbose=0)
    return np.mean((X - X_pred) ** 2, axis=1)


def plot_loss_curve(history, out_path: Path):
    train_loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])

    plt.figure(figsize=(8, 5))
    plt.plot(train_loss, label="Train loss")
    plt.plot(val_loss, label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Autoencoder Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_error_distribution(errors_legit, errors_fraud, t_review, t_block, out_path: Path):
    plt.figure(figsize=(10, 6))
    plt.hist(errors_legit, bins=60, alpha=0.6, density=True, label="Legit (y=0)")
    plt.hist(errors_fraud, bins=60, alpha=0.6, density=True, label="Fraud (y=1)")

    plt.axvline(t_review, linestyle="--", linewidth=2, label=f"Review t={t_review:.4f}")
    plt.axvline(t_block, linestyle="--", linewidth=2, label=f"Block t={t_block:.4f}")

    plt.title("Reconstruction Error Distribution")
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def triage_from_errors(errors: np.ndarray, t_review: float, t_block: float) -> np.ndarray:
    """
    0 = approve (normal)
    1 = review  (anomalous)
    2 = block   (highly anomalous)
    """
    decisions = np.zeros(len(errors), dtype=int)
    decisions[errors >= t_review] = 1
    decisions[errors >= t_block] = 2
    return decisions


def bin_metrics(y_true: np.ndarray, pred_bin: np.ndarray, scores: np.ndarray) -> dict:
    cm = confusion_matrix(y_true, pred_bin)
    tn, fp, fn, tp = cm.ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    return {
        "precision": float(precision_score(y_true, pred_bin, zero_division=0)),
        "recall": float(recall_score(y_true, pred_bin, zero_division=0)),
        "f1": float(f1_score(y_true, pred_bin, zero_division=0)),
        "fpr": fpr,
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "confusion_matrix": cm.tolist(),
        "tn_fp_fn_tp": [int(tn), int(fp), int(fn), int(tp)],
    }


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    print("🔄 Loading datasets...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_datasets()
    print("✔ Loaded.")
    print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)

    # Train ONLY on legitimate
    legit_train_mask = (y_train == 0)
    legit_val_mask = (y_val == 0)

    X_train_legit = X_train[legit_train_mask].values.astype("float32")
    X_val_legit = X_val[legit_val_mask].values.astype("float32")

    print("\nClass distribution (train):")
    print(y_train.value_counts(normalize=True))

    print("\nShapes after filtering legit:")
    print("  X_train_legit:", X_train_legit.shape)
    print("  X_val_legit:  ", X_val_legit.shape)

    input_dim = X_train_legit.shape[1]
    autoencoder = build_autoencoder(input_dim)
    autoencoder.summary()

    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
        verbose=1,
    )

    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1,
    )

    print("\n🚀 Training autoencoder on legitimate transactions only...")
    history = autoencoder.fit(
        X_train_legit,
        X_train_legit,
        validation_data=(X_val_legit, X_val_legit),
        epochs=100,
        batch_size=512,
        callbacks=[early_stop, reduce_lr],
        verbose=1,
    )

    plot_loss_curve(history, AE_LOSS_PLOT_PATH)
    print(f"📈 Saved loss curve to {AE_LOSS_PLOT_PATH}")

    print("\n📊 Computing reconstruction errors...")
    val_legit_errors = reconstruction_errors(autoencoder, X_val_legit)

    # --- Two-threshold policy ---
    # Review: allow 1% of legit to be flagged as "review" (adjustable)
    MAX_FPR_REVIEW = 0.01
    t_review = float(np.percentile(val_legit_errors, 100 * (1 - MAX_FPR_REVIEW)))

    # Block: extreme anomalies only (0.1% legit) (adjustable)
    MAX_FPR_BLOCK = 0.001
    t_block = float(np.percentile(val_legit_errors, 100 * (1 - MAX_FPR_BLOCK)))

    if t_block < t_review:
        # safety: ensure block is stricter
        t_block = t_review

    thresholds = {
        "review": t_review,
        "block": t_block,
        "max_fpr_review": MAX_FPR_REVIEW,
        "max_fpr_block": MAX_FPR_BLOCK,
        "note": "Decisions: approve < review < block based on reconstruction error.",
    }

    with open(AE_THRESHOLDS_PATH, "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"💾 Saved thresholds to {AE_THRESHOLDS_PATH}")
    print(f"   t_review={t_review:.6f} (≈{MAX_FPR_REVIEW*100:.2f}% legit flagged)")
    print(f"   t_block ={t_block:.6f} (≈{MAX_FPR_BLOCK*100:.2f}% legit blocked)")

    # Evaluate on VAL and TEST (full sets)
    X_val_all = X_val.values.astype("float32")
    X_test_all = X_test.values.astype("float32")

    val_errors_all = reconstruction_errors(autoencoder, X_val_all)
    test_errors_all = reconstruction_errors(autoencoder, X_test_all)

    # Binary interpretation for metrics:
    # - "flagged" = review OR block  (errors >= t_review)
    # - "blocked" = block only       (errors >= t_block)
    y_val_arr = y_val.values
    y_test_arr = y_test.values

    val_flagged = (val_errors_all >= t_review).astype(int)
    test_flagged = (test_errors_all >= t_review).astype(int)

    val_blocked = (val_errors_all >= t_block).astype(int)
    test_blocked = (test_errors_all >= t_block).astype(int)

    metrics = {
        "thresholds": thresholds,
        "validation": {
            "flagged_review_or_block": bin_metrics(y_val_arr, val_flagged, val_errors_all),
            "blocked_only": bin_metrics(y_val_arr, val_blocked, val_errors_all),
        },
        "test": {
            "flagged_review_or_block": bin_metrics(y_test_arr, test_flagged, test_errors_all),
            "blocked_only": bin_metrics(y_test_arr, test_blocked, test_errors_all),
        },
        "training": {
            "epochs": int(len(history.history.get("loss", []))),
            "final_train_loss": float(history.history.get("loss", [None])[-1]),
            "final_val_loss": float(history.history.get("val_loss", [None])[-1]),
        },
        "paths": {
            "train": str(TRAIN_PATH),
            "val": str(VAL_PATH),
            "test": str(TEST_PATH),
        },
    }

    with open(AE_METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"💾 Saved metrics to {AE_METRICS_PATH}")

    # Plot distribution on TEST
    errors_legit_test = test_errors_all[y_test_arr == 0]
    errors_fraud_test = test_errors_all[y_test_arr == 1]
    plot_error_distribution(errors_legit_test, errors_fraud_test, t_review, t_block, AE_ERROR_DIST_PLOT_PATH)
    print(f"📈 Saved error distribution plot to {AE_ERROR_DIST_PLOT_PATH}")

    # Save model
    autoencoder.save(AE_MODEL_PATH)
    print(f"💾 Saved autoencoder model to {AE_MODEL_PATH}")

    # Quick triage counts on TEST
    tri_test = triage_from_errors(test_errors_all, t_review, t_block)
    approve = int((tri_test == 0).sum())
    review = int((tri_test == 1).sum())
    block = int((tri_test == 2).sum())

    print("\n✅ Triage counts on TEST")
    print(f"  Approve: {approve}")
    print(f"  Review : {review}")
    print(f"  Block  : {block}")


if __name__ == "__main__":
    main()
