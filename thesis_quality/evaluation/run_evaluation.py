# thesis_quality/evaluation/run_evaluation.py
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

EVAL_SCRIPTS = REPO_ROOT / "scripts" / "eval"
OUT_DIR = REPO_ROOT / "thesis_quality" / "evaluation"
METRICS_DIR = OUT_DIR / "metrics"
PLOTS_DIR = OUT_DIR / "plots"

OUT_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_TEST = REPO_ROOT / "data" / "processed" / "test.csv"
DEFAULT_FEATURES = REPO_ROOT / "artifacts" / "preprocess" / "features.json"
DEFAULT_PREPROCESS = REPO_ROOT / "artifacts" / "preprocess" / "preprocess.joblib"
DEFAULT_XGB = REPO_ROOT / "artifacts" / "models" / "xgb_model.pkl"
DEFAULT_AE_ERRORS = REPO_ROOT / "artifacts" / "ae_errors" / "ae_test_errors.npy"
DEFAULT_AE_THRESH = REPO_ROOT / "artifacts" / "thresholds" / "ae_threshold.txt"

def _run(cmd: list[str]) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)

def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"[COPY] {src} -> {dst}")

def _plot_f1_vs_threshold(
    model_path: Path,
    val_csv: Path,
    out_dir: Path,
) -> None:
    import numpy as np
    import pandas as pd
    import joblib
    import matplotlib.pyplot as plt
    from sklearn.metrics import f1_score

    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = joblib.load(model_path)

    # Load validation data
    val = pd.read_csv(val_csv)
    X_val = val.drop(columns=["Class"])
    y_val = val["Class"]

    # Predict probabilities
    y_proba = model.predict_proba(X_val)[:, 1]

    # Sweep thresholds
    thresholds = np.linspace(0.01, 0.99, 50)
    f1_scores = []

    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        f1_scores.append(f1_score(y_val, preds))

    best_idx = int(np.argmax(f1_scores))
    best_t = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    # Plot
    plt.figure()
    plt.plot(thresholds, f1_scores)
    plt.axvline(best_t, linestyle="--", label=f"Best t={best_t:.2f}, F1={best_f1:.3f}")
    plt.xlabel("Threshold")
    plt.ylabel("F1 score")
    plt.title("F1 vs Threshold (Validation)")
    plt.legend()

    out_path = out_dir / "xgb_f1_vs_threshold_val.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    print(f"[PLOT] Saved {out_path}")


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Thesis-quality evaluation (wrapper).")
    p.add_argument("--test-csv", default=str(DEFAULT_TEST))
    p.add_argument("--features-path", default=str(DEFAULT_FEATURES))
    p.add_argument("--preprocess-joblib", default=str(DEFAULT_PREPROCESS))
    p.add_argument("--xgb-model", default=str(DEFAULT_XGB))
    p.add_argument("--ae-test-errors", default=str(DEFAULT_AE_ERRORS))
    p.add_argument("--ae-threshold", default=str(DEFAULT_AE_THRESH))
    args = p.parse_args()

    protocol = EVAL_SCRIPTS / "run_evaluation_protocol.py"
    compare = REPO_ROOT / "scripts" / "compare_models.py"

    if protocol.exists():
        # Let your main eval script generate everything under docs/figures or artifacts/plots etc.
        _run(
            [
                sys.executable,
                str(protocol),
                "--test-csv",
                str(Path(args.test_csv)),
                "--features-path",
                str(Path(args.features_path)),
                "--preprocess-joblib",
                str(Path(args.preprocess_joblib)),
                "--xgb-model",
                str(Path(args.xgb_model)),
                "--ae-test-errors",
                str(Path(args.ae_test_errors)),
                "--ae-threshold",
                str(Path(args.ae_threshold)),
            ]
        )
    elif compare.exists():
        _run([sys.executable, str(compare)])
    else:
        raise FileNotFoundError("No evaluation driver found: scripts/eval/run_evaluation_protocol.py or scripts/compare_models.py")

    # Copy common outputs into thesis_quality/evaluation
    # Adjust these if your eval scripts write to different locations.
    candidates = [
        REPO_ROOT / "docs" / "figures" / "models" / "04_xgboost" / "xgb_roc_test.png",
        REPO_ROOT / "docs" / "figures" / "models" / "04_xgboost" / "xgb_pr_test.png",
        REPO_ROOT / "docs" / "figures" / "models" / "06_model_comparison" / "roc_test.png",
        REPO_ROOT / "docs" / "figures" / "models" / "06_model_comparison" / "pr_test.png",
        REPO_ROOT / "docs" / "figures" / "models" / "06_model_comparison" / "model_comparison_summary.json",
        REPO_ROOT / "artifacts" / "metrics" / "xgb_metrics.json",
        REPO_ROOT / "artifacts" / "metrics" / "ae_metrics.json",
    ]

    for src in candidates:
        if src.suffix.lower() in [".png"]:
            _copy_if_exists(src, PLOTS_DIR / src.name)
        elif src.suffix.lower() in [".json"]:
            _copy_if_exists(src, METRICS_DIR / src.name)

        # Generate F1 vs threshold plot for thesis (validation set)
    val_csv = REPO_ROOT / "data" / "processed" / "val.csv"

    _plot_f1_vs_threshold(
        model_path=Path(args.xgb_model),
        val_csv=val_csv,
        out_dir=PLOTS_DIR,
    )

    print("[OK] Thesis-quality evaluation outputs collected under:")
    print(f"- {METRICS_DIR}")
    print(f"- {PLOTS_DIR}")

if __name__ == "__main__":
    main()
