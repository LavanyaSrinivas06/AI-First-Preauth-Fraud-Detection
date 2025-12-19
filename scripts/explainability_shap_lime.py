from __future__ import annotations
#!/usr/bin/env python3
"""
FPN-10 | Explainability (SHAP + LIME)

Goal
- SHAP: global + local interpretability for XGBoost
- LIME: sample-level explanations for flagged fraud examples

Outputs
- artifacts/explainability/shap/
- artifacts/explainability/lime/
- docs/explainability_report.md (we will generate later)
"""



import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

# Matplotlib is used for saving plots (no GUI needed)
import matplotlib.pyplot as plt

# Explainability libraries
import shap
from lime.lime_tabular import LimeTabularExplainer


# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class ExplainabilityConfig:
    # Input artifacts
    model_path: Path = Path("artifacts/xgb_model.pkl")
    preprocess_path: Path = Path("artifacts/preprocess.joblib")
    features_meta_path: Path = Path("artifacts/features.json")

    # Data
    test_csv_path: Path = Path("data/processed/test.csv")
    val_csv_path: Path = Path("data/processed/val.csv")
    train_csv_path: Path = Path("data/processed/train.csv")

    target_col: str = "Class"

    # Output folders
    shap_dir: Path = Path("artifacts/explainability/shap")
    lime_dir: Path = Path("artifacts/explainability/lime")

    # How many examples to explain
    n_force_plots: int = 2
    n_lime_explanations: int = 5

    # Sampling (to keep SHAP manageable with 7k+ features)
    shap_sample_size: int = 2000   # global plots computed on sample
    lime_background_size: int = 2000  # background for LIME explainer


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_dirs(cfg: ExplainabilityConfig) -> None:
    cfg.shap_dir.mkdir(parents=True, exist_ok=True)
    cfg.lime_dir.mkdir(parents=True, exist_ok=True)


def _load_features_meta(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_dataset(csv_path: Path, target_col: str) -> Tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {csv_path}")
    y = df[target_col].astype(int).values
    X_raw = df.drop(columns=[target_col])
    return X_raw, y


def _safe_feature_names(preprocess) -> np.ndarray:
    # ColumnTransformer in sklearn supports get_feature_names_out in newer versions
    try:
        names = preprocess.get_feature_names_out()
        return np.array(names, dtype=str)
    except Exception:
        return np.array([], dtype=str)


def _pick_fraud_indices(y: np.ndarray) -> np.ndarray:
    return np.where(y == 1)[0]


# -----------------------------
# SHAP
# -----------------------------
#!/usr/bin/env python3
"""
FPN-10 | Explainability (SHAP + LIME)

Goal
- SHAP: global + local interpretability for XGBoost
- LIME: sample-level explanations for flagged fraud examples

Outputs
- artifacts/explainability/shap/
- artifacts/explainability/lime/
- docs/explainability_report.md (we will generate later)
"""


import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

# Matplotlib is used for saving plots (no GUI needed)
import matplotlib.pyplot as plt

# Explainability libraries
import shap
from lime.lime_tabular import LimeTabularExplainer


# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class ExplainabilityConfig:
    # Input artifacts
    model_path: Path = Path("artifacts/xgb_model.pkl")
    preprocess_path: Path = Path("artifacts/preprocess.joblib")
    features_meta_path: Path = Path("artifacts/features.json")

    # Data
    test_csv_path: Path = Path("data/processed/test.csv")
    val_csv_path: Path = Path("data/processed/val.csv")
    train_csv_path: Path = Path("data/processed/train.csv")

    target_col: str = "Class"

    # Output folders
    shap_dir: Path = Path("artifacts/explainability/shap")
    lime_dir: Path = Path("artifacts/explainability/lime")

    # How many examples to explain
    n_force_plots: int = 2
    n_lime_explanations: int = 5

    # Sampling (to keep SHAP manageable with 7k+ features)
    shap_sample_size: int = 2000   # global plots computed on sample
    lime_background_size: int = 2000  # background for LIME explainer


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_dirs(cfg: ExplainabilityConfig) -> None:
    cfg.shap_dir.mkdir(parents=True, exist_ok=True)
    cfg.lime_dir.mkdir(parents=True, exist_ok=True)


def _load_features_meta(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_dataset(csv_path: Path, target_col: str) -> Tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {csv_path}")
    y = df[target_col].astype(int).values
    X_raw = df.drop(columns=[target_col])
    return X_raw, y


def _safe_feature_names(preprocess) -> np.ndarray:
    # ColumnTransformer in sklearn supports get_feature_names_out in newer versions
    try:
        names = preprocess.get_feature_names_out()
        return np.array(names, dtype=str)
    except Exception:
        return np.array([], dtype=str)


def _pick_fraud_indices(y: np.ndarray) -> np.ndarray:
    return np.where(y == 1)[0]


# -----------------------------
# SHAP
# -----------------------------
def run_shap(
    model,
    X_sample: np.ndarray,
    feature_names: np.ndarray,
    out_dir: Path,
    force_indices_in_sample: List[int],
) -> Dict[str, Any]:
    """
    Generates:
    - SHAP summary plot (dot)
    - SHAP bar plot (global importance)
    - SHAP force plots for selected samples
    """
    results: Dict[str, Any] = {"shap": {}}

    # --- IMPORTANT FIX ---
    # SHAP works more reliably with the underlying Booster than with sklearn wrapper.
    booster = model.get_booster()
    explainer = shap.TreeExplainer(booster)

    shap_values = explainer.shap_values(X_sample)

    # For binary classification, shap_values may be:
    # - np.ndarray of shape (n, d)
    # - OR list length 2, where index 1 corresponds to the positive class
    if isinstance(shap_values, list):
        # use positive class explanations
        shap_values_to_use = shap_values[1]
    else:
        shap_values_to_use = shap_values

    # expected_value may also be list for some setups
    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)) and not np.isscalar(expected_value):
        expected_value_to_use = expected_value[1] if len(expected_value) > 1 else expected_value[0]
    else:
        expected_value_to_use = expected_value

    # Save SHAP values
    shap_values_path = out_dir / "shap_values.npy"
    np.save(shap_values_path, shap_values_to_use)
    results["shap"]["shap_values_npy"] = str(shap_values_path)

    # Summary (beeswarm)
    summary_path = out_dir / "shap_summary.png"
    plt.figure()
    shap.summary_plot(
        shap_values_to_use,
        X_sample,
        feature_names=feature_names,
        show=False,
        max_display=30,
    )
    plt.savefig(summary_path, bbox_inches="tight", dpi=200)
    plt.close()
    results["shap"]["summary_plot"] = str(summary_path)

    # Bar plot (global importance)
    bar_path = out_dir / "shap_bar.png"
    plt.figure()
    shap.summary_plot(
        shap_values_to_use,
        X_sample,
        feature_names=feature_names,
        show=False,
        plot_type="bar",
        max_display=30,
    )
    plt.savefig(bar_path, bbox_inches="tight", dpi=200)
    plt.close()
    results["shap"]["bar_plot"] = str(bar_path)

    # Force plots (local)
    force_paths: List[str] = []
    for i, idx in enumerate(force_indices_in_sample):
        force_path = out_dir / f"shap_force_{i}.png"
        fig = shap.force_plot(
            expected_value_to_use,
            shap_values_to_use[idx],
            X_sample[idx],
            feature_names=feature_names,
            matplotlib=True,
        )
        fig.savefig(force_path, bbox_inches="tight", dpi=200)
        plt.close(fig)
        force_paths.append(str(force_path))

    results["shap"]["force_plots"] = force_paths
    return results

# -----------------------------
# LIME
# -----------------------------
def run_lime(
    model,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    explain_indices: List[int],
    feature_names: np.ndarray,
    out_dir: Path,
) -> Dict[str, Any]:
    """
    Generates HTML explanations for selected samples.
    """
    results: Dict[str, Any] = {"lime": {"html_explanations": []}}

    explainer = LimeTabularExplainer(
        training_data=X_background,
        feature_names=feature_names.tolist(),
        class_names=["Legit", "Fraud"],
        mode="classification",
        discretize_continuous=True,
    )

    for i, idx in enumerate(explain_indices):
        exp = explainer.explain_instance(
            X_explain[idx],
            model.predict_proba,
            num_features=10,
        )
        html_path = out_dir / f"lime_explanation_{i}.html"
        exp.save_to_file(str(html_path))
        results["lime"]["html_explanations"].append(
            {"sample_index": int(idx), "file": str(html_path)}
        )

    # Save metadata JSON
    meta_path = out_dir / "lime_results.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(results["lime"]["html_explanations"], f, indent=2)
    results["lime"]["metadata_json"] = str(meta_path)

    return results


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    cfg = ExplainabilityConfig()
    _ensure_dirs(cfg)

    print("🔄 Loading model + preprocessing pipeline...")
    preprocess = joblib.load(cfg.preprocess_path)
    model = joblib.load(cfg.model_path)
    print("🔄 Loading datasets...")
    X_test_raw, y_test = _load_dataset(cfg.test_csv_path, cfg.target_col)

    df_test = pd.read_csv(cfg.test_csv_path)
    y_test = df_test[cfg.target_col].astype(int).values
    # test.csv is already processed (encoded + scaled), so no transform here
    X_test = df_test.drop(columns=[cfg.target_col]).values
    feature_names = df_test.drop(columns=[cfg.target_col]).columns.astype(str).to_numpy()

    if feature_names.size == 0:
        # Fallback: try features.json if pipeline doesn't expose names
        meta = _load_features_meta(cfg.features_meta_path)
        if "feature_names" in meta and isinstance(meta["feature_names"], list):
            feature_names = np.array(meta["feature_names"], dtype=str)

    if feature_names.size == 0:
        raise RuntimeError("Could not resolve feature names (pipeline + features.json both failed).")

    # Sample to keep SHAP and LIME runtime reasonable
    rng = np.random.default_rng(42)
    n_test = X_test.shape[0]
    sample_size = min(cfg.shap_sample_size, n_test)
    sample_idx = rng.choice(n_test, size=sample_size, replace=False)

    X_sample = X_test[sample_idx]

    # Choose fraud examples for force plots:
    # - find fraud rows in the full test set
    fraud_indices_full = _pick_fraud_indices(y_test)
    if fraud_indices_full.size == 0:
        print("⚠️ No fraud samples found in test set. Force plots will use random samples instead.")
        force_indices_in_sample = list(range(min(cfg.n_force_plots, X_sample.shape[0])))
    else:
        # map fraud indices to the sample index space if possible
        fraud_in_sample = [i for i, idx in enumerate(sample_idx) if idx in set(fraud_indices_full)]
        if len(fraud_in_sample) == 0:
            # fallback: pick top predicted fraud probability in the sample
            proba = model.predict_proba(X_sample)[:, 1]
            topk = np.argsort(proba)[::-1][: cfg.n_force_plots]
            force_indices_in_sample = topk.tolist()
        else:
            force_indices_in_sample = fraud_in_sample[: cfg.n_force_plots]

    print("✅ Running SHAP (global + local)...")
    shap_results = run_shap(
        model=model,
        X_sample=X_sample,
        feature_names=feature_names,
        out_dir=cfg.shap_dir,
        force_indices_in_sample=force_indices_in_sample,
    )

    # For LIME background, reuse the same sample
    background_size = min(cfg.lime_background_size, X_sample.shape[0])
    background_idx = rng.choice(X_sample.shape[0], size=background_size, replace=False)
    X_background = X_sample[background_idx]

    # Explain top predicted fraud samples (or fraud samples if present)
    proba_test = model.predict_proba(X_test)[:, 1]
    top_pred_indices = np.argsort(proba_test)[::-1][: cfg.n_lime_explanations].tolist()

    print("✅ Running LIME (local explanations)...")
    lime_results = run_lime(
        model=model,
        X_background=X_background,
        X_explain=X_test,
        explain_indices=top_pred_indices,
        feature_names=feature_names,
        out_dir=cfg.lime_dir,
    )

    # Save one combined metadata file for dashboard use
    combined = {
        "generated_at": _now_utc_iso(),
        "inputs": {
            "model_path": str(cfg.model_path),
            "preprocess_path": str(cfg.preprocess_path),
            "test_csv_path": str(cfg.test_csv_path),
        },
        **shap_results,
        **lime_results,
    }

    combined_path = Path("artifacts/explainability/explainability_index.json")
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    print("✅ Explainability generation complete.")
    print(f"  SHAP outputs: {cfg.shap_dir}")
    print(f"  LIME outputs: {cfg.lime_dir}")
    print(f"  Index JSON:   {combined_path}")


if __name__ == "__main__":
    main()



# -----------------------------
# LIME
# -----------------------------
def run_lime(
    model,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    explain_indices: List[int],
    feature_names: np.ndarray,
    out_dir: Path,
) -> Dict[str, Any]:
    """
    Generates HTML explanations for selected samples.
    """
    results: Dict[str, Any] = {"lime": {"html_explanations": []}}

    explainer = LimeTabularExplainer(
        training_data=X_background,
        feature_names=feature_names.tolist(),
        class_names=["Legit", "Fraud"],
        mode="classification",
        discretize_continuous=True,
    )

    for i, idx in enumerate(explain_indices):
        exp = explainer.explain_instance(
            X_explain[idx],
            model.predict_proba,
            num_features=10,
        )
        html_path = out_dir / f"lime_explanation_{i}.html"
        exp.save_to_file(str(html_path))
        results["lime"]["html_explanations"].append(
            {"sample_index": int(idx), "file": str(html_path)}
        )

    # Save metadata JSON
    meta_path = out_dir / "lime_results.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(results["lime"]["html_explanations"], f, indent=2)
    results["lime"]["metadata_json"] = str(meta_path)

    return results


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    cfg = ExplainabilityConfig()
    _ensure_dirs(cfg)

    print("🔄 Loading model + preprocessing pipeline...")
    preprocess = joblib.load(cfg.preprocess_path)
    model = joblib.load(cfg.model_path)
    print("🔄 Loading datasets...")
    X_test_raw, y_test = _load_dataset(cfg.test_csv_path, cfg.target_col)

    df_test = pd.read_csv(cfg.test_csv_path)
    y_test = df_test[cfg.target_col].astype(int).values
    # test.csv is already processed (encoded + scaled), so no transform here
    X_test = df_test.drop(columns=[cfg.target_col]).values
    feature_names = df_test.drop(columns=[cfg.target_col]).columns.astype(str).to_numpy()

    if feature_names.size == 0:
        # Fallback: try features.json if pipeline doesn't expose names
        meta = _load_features_meta(cfg.features_meta_path)
        if "feature_names" in meta and isinstance(meta["feature_names"], list):
            feature_names = np.array(meta["feature_names"], dtype=str)

    if feature_names.size == 0:
        raise RuntimeError("Could not resolve feature names (pipeline + features.json both failed).")

    # Sample to keep SHAP and LIME runtime reasonable
    rng = np.random.default_rng(42)
    n_test = X_test.shape[0]
    sample_size = min(cfg.shap_sample_size, n_test)
    sample_idx = rng.choice(n_test, size=sample_size, replace=False)

    X_sample = X_test[sample_idx]

    # Choose fraud examples for force plots:
    # - find fraud rows in the full test set
    fraud_indices_full = _pick_fraud_indices(y_test)
    if fraud_indices_full.size == 0:
        print("⚠️ No fraud samples found in test set. Force plots will use random samples instead.")
        force_indices_in_sample = list(range(min(cfg.n_force_plots, X_sample.shape[0])))
    else:
        # map fraud indices to the sample index space if possible
        fraud_in_sample = [i for i, idx in enumerate(sample_idx) if idx in set(fraud_indices_full)]
        if len(fraud_in_sample) == 0:
            # fallback: pick top predicted fraud probability in the sample
            proba = model.predict_proba(X_sample)[:, 1]
            topk = np.argsort(proba)[::-1][: cfg.n_force_plots]
            force_indices_in_sample = topk.tolist()
        else:
            force_indices_in_sample = fraud_in_sample[: cfg.n_force_plots]

    print("✅ Running SHAP (global + local)...")
    shap_results = run_shap(
        model=model,
        X_sample=X_sample,
        feature_names=feature_names,
        out_dir=cfg.shap_dir,
        force_indices_in_sample=force_indices_in_sample,
    )

    # For LIME background, reuse the same sample
    background_size = min(cfg.lime_background_size, X_sample.shape[0])
    background_idx = rng.choice(X_sample.shape[0], size=background_size, replace=False)
    X_background = X_sample[background_idx]

    # Explain top predicted fraud samples (or fraud samples if present)
    proba_test = model.predict_proba(X_test)[:, 1]
    top_pred_indices = np.argsort(proba_test)[::-1][: cfg.n_lime_explanations].tolist()

    print("✅ Running LIME (local explanations)...")
    lime_results = run_lime(
        model=model,
        X_background=X_background,
        X_explain=X_test,
        explain_indices=top_pred_indices,
        feature_names=feature_names,
        out_dir=cfg.lime_dir,
    )

    # Save one combined metadata file for dashboard use
    combined = {
        "generated_at": _now_utc_iso(),
        "inputs": {
            "model_path": str(cfg.model_path),
            "preprocess_path": str(cfg.preprocess_path),
            "test_csv_path": str(cfg.test_csv_path),
        },
        **shap_results,
        **lime_results,
    }

    combined_path = Path("artifacts/explainability/explainability_index.json")
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    print("✅ Explainability generation complete.")
    print(f"  SHAP outputs: {cfg.shap_dir}")
    print(f"  LIME outputs: {cfg.lime_dir}")
    print(f"  Index JSON:   {combined_path}")


if __name__ == "__main__":
    main()
