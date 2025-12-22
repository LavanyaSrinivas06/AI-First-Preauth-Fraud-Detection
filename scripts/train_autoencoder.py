#!/usr/bin/env python3
"""
Train an unsupervised Autoencoder on legitimate (non-fraud) transactions.

Ticket: FPN-8
Phase: Phase 2 — Model Training & Evaluation

Inputs:
- data/processed/train.csv
- data/processed/val.csv
- data/processed/test.csv

Outputs:
- artifacts/autoencoder_model.h5
- artifacts/ae_threshold.txt
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
# Paths and directories
# -------------------------------------------------------------------

# scripts/ → go 1 level up → repo root
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed"
ARTIFACT_DIR = ROOT / "artifacts"
PLOT_DIR = ARTIFACT_DIR / "plots"

ARTIFACT_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)

TRAIN_PATH = DATA_DIR / "train_nosmote.csv"
VAL_PATH   = DATA_DIR / "val_nosmote.csv"
TEST_PATH  = DATA_DIR / "test_nosmote.csv"

AE_MODEL_PATH = ARTIFACT_DIR / "autoencoder_model.h5"
AE_THRESHOLD_PATH = ARTIFACT_DIR / "ae_threshold.txt"
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
    """Load train, validation, and test sets from CSV."""
    if not (TRAIN_PATH.exists() and VAL_PATH.exists() and TEST_PATH.exists()):
        raise FileNotFoundError("One or more processed CSV files are missing.")

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
    """Build a simple dense Autoencoder."""
    model = models.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(64, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(input_dim, activation="linear"),
        ]
    )
    opt = tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)
    model.compile(optimizer=opt, loss="mse")

    return model


def reconstruction_errors(model: models.Model, X: np.ndarray) -> np.ndarray:
    """Compute per-row MSE reconstruction errors."""
    X = np.asarray(X)
    X_pred = model.predict(X, verbose=0)

    if X.shape != X_pred.shape:
        raise ValueError(f"Shape mismatch: X={X.shape}, X_pred={X_pred.shape}")

    return np.mean((X - X_pred) ** 2, axis=1)


def plot_loss_curve(history, out_path: Path):
    """Plot training vs validation loss and save as PNG."""
    train_loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])

    plt.figure(figsize=(8, 5))
    plt.plot(train_loss, label="Train loss")
    plt.plot(val_loss, label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Autoencoder Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_error_distribution(
    errors_legit: np.ndarray,
    errors_fraud: np.ndarray,
    threshold: float,
    out_path: Path,
):
    """Plot reconstruction error distributions for legit vs fraud."""
    plt.figure(figsize=(10, 6))

    plt.hist(errors_legit, bins=50, alpha=0.6, density=True, label="Legit (y=0)")
    plt.hist(errors_fraud, bins=50, alpha=0.6, density=True, label="Fraud (y=1)")

    plt.axvline(
        threshold,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Threshold = {threshold:.4f}",
    )

    plt.title("Autoencoder Reconstruction Error Distribution", fontsize=14)
    plt.xlabel("Reconstruction Error", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# -------------------------------------------------------------------
# Main autoencoder training & evaluation
# -------------------------------------------------------------------

def main():
    print("🔄 Loading datasets...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_datasets()
    print("✔ Loaded.")
    print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)

    # Filter only legitimate samples for training and validation thresholding
    legit_train_mask = y_train == 0
    legit_val_mask = y_val == 0

    X_train_legit = X_train[legit_train_mask].values
    X_val_legit = X_val[legit_val_mask].values

    X_train_legit = X_train_legit.astype("float32")
    X_val_legit   = X_val_legit.astype("float32")


    print("\nClass distribution (train):")
    print(y_train.value_counts(normalize=True))

    print("\nShapes after filtering legit:")
    print("  X_train_legit:", X_train_legit.shape)
    print("  X_val_legit:  ", X_val_legit.shape)

    # Build autoencoder
    input_dim = X_train_legit.shape[1]
    autoencoder = build_autoencoder(input_dim)
    autoencoder.summary()

    # Train with early stopping
    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )

    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
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

    # Save loss curve
    plot_loss_curve(history, AE_LOSS_PLOT_PATH)
    print(f"📈 Saved loss curve to {AE_LOSS_PLOT_PATH}")

    X_test_arr = X_test.values.astype("float32")
    test_errors = reconstruction_errors(autoencoder, X_test_arr)


    # Compute reconstruction errors
    print("\n📊 Computing reconstruction errors...")
    val_legit_errors = reconstruction_errors(autoencoder, X_val_legit)
    test_errors = reconstruction_errors(autoencoder, X_test.values)

    print("Val legit errors:")
    print("  mean:", float(val_legit_errors.mean()))
    print("  std: ", float(val_legit_errors.std()))
    print("Test errors:")
    print("  mean:", float(test_errors.mean()))
    print("  std: ", float(test_errors.std()))

    # Choose threshold from validation legit errors by target false-positive rate
    # Example: allow 0.5% of legit to be flagged (adjust to your business tolerance)
    TARGET_FPR = 0.005  # 0.5%
    percentile = 100 * (1 - TARGET_FPR) # = 99.5
    threshold = float(np.percentile(val_legit_errors, percentile))

    print(f"\nChosen threshold ({percentile:.2f}th percentile of legit val errors): {threshold}")

    # Classify test samples based on threshold
    y_test_array = y_test.values
    y_scores = test_errors  # higher = more anomalous
    y_pred = (test_errors >= threshold).astype(int)

    # Metrics
    precision = float(precision_score(y_test_array, y_pred, zero_division=0))
    recall = float(recall_score(y_test_array, y_pred, zero_division=0))  # TPR
    f1 = float(f1_score(y_test_array, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_test_array, y_scores))
    pr_auc = float(average_precision_score(y_test_array, y_scores))
    cm = confusion_matrix(y_test_array, y_pred)

    tn, fp, fn, tp = cm.ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    print("\n📊 Autoencoder Performance on Test Set")
    print("--------------------------------------")
    print("Threshold:          ", threshold)
    print("Precision:          ", precision)
    print("Recall (TPR):       ", recall)
    print("False Positive Rate:", fpr)
    print("F1-score:           ", f1)
    print("ROC-AUC:            ", roc_auc)
    print("PR-AUC:             ", pr_auc)
    print("\nConfusion Matrix:\n", cm)

    # Plot error distributions
    errors_legit = test_errors[y_test_array == 0]
    errors_fraud = test_errors[y_test_array == 1]
    plot_error_distribution(
        errors_legit,
        errors_fraud,
        threshold,
        AE_ERROR_DIST_PLOT_PATH,
    )
    print(f"📈 Saved error distribution plot to {AE_ERROR_DIST_PLOT_PATH}")

    # Save model
    autoencoder.save(AE_MODEL_PATH)
    print(f"💾 Saved autoencoder model to {AE_MODEL_PATH}")

    # Save threshold
    with open(AE_THRESHOLD_PATH, "w") as f:
        f.write(str(threshold))
    print(f"💾 Saved threshold to {AE_THRESHOLD_PATH}")

    # Save metrics as JSON
    metrics = {
        "threshold": threshold,
        "test": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fpr": fpr,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "confusion_matrix": cm.tolist(),
            "tn_fp_fn_tp": [int(tn), int(fp), int(fn), int(tp)],
        },
        "training": {
            "epochs": len(history.history.get("loss", [])),
            "final_train_loss": float(history.history.get("loss", [None])[-1]),
            "final_val_loss": float(history.history.get("val_loss", [None])[-1]),
        },
    }

    with open(AE_METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"💾 Saved metrics to {AE_METRICS_PATH}")

    # Simple acceptance check (logged, not a hard fail)
    print("\n✅ Acceptance Criteria Check (on TEST):")
    print(f"  TPR (recall) >= 0.85 ? {'YES' if recall >= 0.85 else 'NO'}")
    print(f"  FPR <= 0.10       ? {'YES' if fpr <= 0.10 else 'NO'}")


if __name__ == "__main__":
    main()
