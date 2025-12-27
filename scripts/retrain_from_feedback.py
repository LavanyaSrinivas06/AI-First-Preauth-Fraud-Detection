# scripts/retrain_from_feedback.py
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import joblib
import pandas as pd
from xgboost import XGBClassifier

ART_DIR = Path("artifacts")
DATA_DIR = Path("data/processed")
DOCS_DIR = Path("docs")

TRAIN_PATH = DATA_DIR / "train.csv"
VAL_PATH = DATA_DIR / "val.csv"
TEST_PATH = DATA_DIR / "test.csv"
FEEDBACK_PATH = DATA_DIR / "feedback_labeled.csv"

OUT_MODEL = ART_DIR / "xgb_model_feedback.pkl"
OUT_METRICS = ART_DIR / "xgb_feedback_metrics.json"
OUT_REPORT = DOCS_DIR / "feedback_retraining_report.md"


def _load_schema_from_train(train_df: pd.DataFrame) -> List[str]:
    if "Class" not in train_df.columns:
        raise ValueError("train.csv must contain Class")
    feats = [c for c in train_df.columns if c != "Class"]
    if len(feats) != 102:
        raise ValueError(f"Expected 102 features in train.csv, got {len(feats)}")
    return feats


def _prep_feedback(feedback_df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    # Expect feedback_labeled.csv has either `label` or `Class`
    if "label" in feedback_df.columns and "Class" not in feedback_df.columns:
        feedback_df = feedback_df.rename(columns={"label": "Class"})

    if "Class" not in feedback_df.columns:
        raise ValueError("feedback_labeled.csv must contain label or Class column")

    # Keep only schema features + Class
    keep_cols = [c for c in features if c in feedback_df.columns]
    if len(keep_cols) != len(features):
        missing = [c for c in features if c not in feedback_df.columns]
        raise ValueError(f"feedback_labeled.csv missing required features: {missing[:10]}")

    fb = feedback_df[keep_cols + ["Class"]].copy()
    fb["Class"] = fb["Class"].astype(int)
    return fb


def _fit_xgb(X: pd.DataFrame, y: pd.Series) -> XGBClassifier:
    # Keep it simple + stable (no re-tuning needed for thesis loop)
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=4,
        random_state=42,
    )
    model.fit(X, y)
    return model


def _eval(model: XGBClassifier, df: pd.DataFrame, features: List[str]) -> dict:
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

    X = df[features]
    y = df["Class"].astype(int)
    p = model.predict_proba(X)[:, 1]
    pred = (p >= 0.5).astype(int)

    out = {
        "rows": int(len(df)),
        "fraud_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, p)) if y.nunique() > 1 else None,
        "pr_auc": float(average_precision_score(y, p)) if y.nunique() > 1 else None,
        "f1@0.5": float(f1_score(y, pred)) if y.nunique() > 1 else None,
    }
    return out


def main() -> None:
    ART_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(TRAIN_PATH)
    val = pd.read_csv(VAL_PATH)
    test = pd.read_csv(TEST_PATH)

    features = _load_schema_from_train(train)

    if not FEEDBACK_PATH.exists():
        raise SystemExit(f"Missing {FEEDBACK_PATH}. Run build_feedback_dataset.py first.")

    fb_raw = pd.read_csv(FEEDBACK_PATH)
    fb = _prep_feedback(fb_raw, features)

    # Append feedback to train (basic dedupe by exact feature row + Class)
    base = train[features + ["Class"]].copy()
    combined = pd.concat([base, fb], ignore_index=True).drop_duplicates()

    model = _fit_xgb(combined[features], combined["Class"].astype(int))

    metrics = {
        "train_base_rows": int(len(base)),
        "feedback_rows": int(len(fb)),
        "combined_rows": int(len(combined)),
        "val": _eval(model, val, features),
        "test": _eval(model, test, features),
        "model_path": str(OUT_MODEL),
    }

    joblib.dump(model, OUT_MODEL)
    OUT_METRICS.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    OUT_REPORT.write_text(
        "\n".join(
            [
                "# Feedback Retraining Report",
                "",
                f"- Base train rows: {metrics['train_base_rows']}",
                f"- Feedback rows added: {metrics['feedback_rows']}",
                f"- Combined rows used: {metrics['combined_rows']}",
                "",
                "## Validation",
                f"- ROC-AUC: {metrics['val']['roc_auc']}",
                f"- PR-AUC: {metrics['val']['pr_auc']}",
                f"- F1@0.5: {metrics['val']['f1@0.5']}",
                "",
                "## Test",
                f"- ROC-AUC: {metrics['test']['roc_auc']}",
                f"- PR-AUC: {metrics['test']['pr_auc']}",
                f"- F1@0.5: {metrics['test']['f1@0.5']}",
                "",
                "## Artifacts",
                f"- Model: `{OUT_MODEL}`",
                f"- Metrics: `{OUT_METRICS}`",
                "",
                "Notes: This retraining run demonstrates the proposal-aligned feedback loop (human labels → dataset → model update).",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Saved: {OUT_MODEL}")
    print(f"Saved: {OUT_METRICS}")
    print(f"Saved: {OUT_REPORT}")


if __name__ == "__main__":
    main()
