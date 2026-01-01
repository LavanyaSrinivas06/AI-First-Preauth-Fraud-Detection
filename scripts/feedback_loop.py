# scripts/feedback_loop.py
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import pandas as pd
from xgboost import XGBClassifier


# -----------------------------
# Paths (single source of truth)
# -----------------------------
REPO_ROOT = Path(".")
SQLITE_PATH = REPO_ROOT / "artifacts" / "stores" / "inference_store.sqlite"

TRAIN_PATH = REPO_ROOT / "data" / "processed" / "train.csv"
VAL_PATH = REPO_ROOT / "data" / "processed" / "val.csv"
TEST_PATH = REPO_ROOT / "data" / "processed" / "test.csv"

FEEDBACK_OUT = REPO_ROOT / "data" / "processed" / "feedback_labeled.csv"

OUT_MODEL = REPO_ROOT / "artifacts" / "models" / "xgb_model_feedback.pkl"
OUT_METRICS = REPO_ROOT / "artifacts" / "metrics" / "xgb_feedback_metrics.json"
OUT_REPORT = REPO_ROOT / "docs" / "feedback_retraining_report.md"


# -----------------------------
# Helpers
# -----------------------------
def _connect(db: Path) -> sqlite3.Connection:
    if not db.exists():
        raise FileNotFoundError(f"DB not found: {db}")
    con = sqlite3.connect(str(db))
    con.row_factory = sqlite3.Row
    return con


def _ensure_dirs() -> None:
    FEEDBACK_OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    OUT_METRICS.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)


def _read_json(path_str: str) -> Dict[str, Any]:
    p = Path(path_str)
    if not p.exists():
        # If stored as relative, try repo root
        p2 = REPO_ROOT / path_str
        if p2.exists():
            p = p2
    if not p.exists():
        raise FileNotFoundError(f"Feature snapshot missing: {path_str}")
    return json.loads(p.read_text(encoding="utf-8"))


def _load_schema_from_train(train_df: pd.DataFrame) -> List[str]:
    if "Class" not in train_df.columns:
        raise ValueError("train.csv must contain Class")
    feats = [c for c in train_df.columns if c != "Class"]
    if len(feats) != 102:
        raise ValueError(f"Expected 102 features in train.csv, got {len(feats)}")
    return feats


def _fit_xgb(X: pd.DataFrame, y: pd.Series) -> XGBClassifier:
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


def _eval(model: XGBClassifier, df: pd.DataFrame, features: List[str]) -> Dict[str, Any]:
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

    X = df[features]
    y = df["Class"].astype(int)
    p = model.predict_proba(X)[:, 1]
    pred = (p >= 0.5).astype(int)

    return {
        "rows": int(len(df)),
        "fraud_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, p)) if y.nunique() > 1 else None,
        "pr_auc": float(average_precision_score(y, p)) if y.nunique() > 1 else None,
        "f1@0.5": float(f1_score(y, pred)) if y.nunique() > 1 else None,
    }


# -----------------------------
# Step A: Backfill missing feature_path
# -----------------------------
def backfill_feature_paths(db: Path) -> int:
    """
    For closed + labeled reviews with missing reviews.feature_path,
    try to copy decisions.feature_path using payload_hash match.
    Returns number of updated review rows.
    """
    con = _connect(db)
    try:
        cur = con.cursor()

        # Find closed labelled reviews with missing feature_path
        cur.execute(
            """
            SELECT id, payload_hash
            FROM reviews
            WHERE status='closed'
              AND analyst_decision IS NOT NULL
              AND (feature_path IS NULL OR TRIM(feature_path) = '')
            """
        )
        missing = cur.fetchall()
        if not missing:
            return 0

        updated = 0
        for row in missing:
            review_id = row["id"]
            payload_hash = row["payload_hash"]
            if not payload_hash:
                continue

            cur.execute(
                """
                SELECT feature_path
                FROM decisions
                WHERE payload_hash = ?
                ORDER BY created DESC
                LIMIT 1
                """,
                (payload_hash,),
            )
            hit = cur.fetchone()
            if not hit:
                continue

            fp = hit["feature_path"]
            if not fp or str(fp).strip() == "":
                continue

            cur.execute(
                """
                UPDATE reviews
                SET feature_path = ?
                WHERE id = ?
                """,
                (fp, review_id),
            )
            if cur.rowcount > 0:
                updated += 1

        con.commit()
        return updated
    finally:
        con.close()


# -----------------------------
# Step B: Export feedback dataset
# -----------------------------
def export_feedback_dataset(db: Path, features: List[str]) -> pd.DataFrame:
    """
    Export closed labelled reviews -> 102 feature rows + label + metadata.
    label mapping:
      BLOCK   -> 1
      APPROVE -> 0
    """
    con = _connect(db)
    try:
        df = pd.read_sql_query(
            """
            SELECT id, analyst_decision, analyst, notes, updated, feature_path
            FROM reviews
            WHERE status='closed'
              AND analyst_decision IS NOT NULL
            ORDER BY updated DESC
            """,
            con,
        )
    finally:
        con.close()

    if df.empty:
        raise SystemExit("No closed labelled reviews found. Close reviews in dashboard first.")

    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        decision = str(r["analyst_decision"]).upper().strip()
        if decision not in {"APPROVE", "BLOCK"}:
            continue

        label = 1 if decision == "BLOCK" else 0
        feature_path = r.get("feature_path")

        if not feature_path or str(feature_path).strip() == "":
            # Skip rows without feature snapshot (should be rare after backfill)
            continue

        feats = _read_json(str(feature_path))

        # enforce schema
        missing = [c for c in features if c not in feats]
        if missing:
            raise ValueError(f"Snapshot {feature_path} missing features: {missing[:10]}")

        row = {c: feats[c] for c in features}
        row["label"] = label
        row["review_id"] = r["id"]
        row["analyst"] = r.get("analyst")
        row["notes"] = r.get("notes")
        row["updated"] = r.get("updated")
        rows.append(row)

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        raise SystemExit("No rows exported. Check feature_path availability and snapshots.")

    out_df.to_csv(FEEDBACK_OUT, index=False)
    return out_df


# -----------------------------
# Step C: Retrain
# -----------------------------
def retrain_with_feedback(features: List[str]) -> Dict[str, Any]:
    train = pd.read_csv(TRAIN_PATH)
    val = pd.read_csv(VAL_PATH)
    test = pd.read_csv(TEST_PATH)

    if not FEEDBACK_OUT.exists():
        raise SystemExit(f"Missing {FEEDBACK_OUT}. Run export step first.")

    fb_raw = pd.read_csv(FEEDBACK_OUT)

    # standardize label column
    if "label" in fb_raw.columns and "Class" not in fb_raw.columns:
        fb_raw = fb_raw.rename(columns={"label": "Class"})

    keep_cols = [c for c in features if c in fb_raw.columns]
    if len(keep_cols) != len(features):
        missing = [c for c in features if c not in fb_raw.columns]
        raise ValueError(f"feedback_labeled.csv missing required features: {missing[:10]}")

    fb = fb_raw[keep_cols + ["Class"]].copy()
    fb["Class"] = fb["Class"].astype(int)

    base = train[features + ["Class"]].copy()
    combined = pd.concat([base, fb], ignore_index=True).drop_duplicates()

    model = _fit_xgb(combined[features], combined["Class"].astype(int))

    metrics = {
        "db_path": str(SQLITE_PATH),
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
                f"- DB used: `{SQLITE_PATH}`",
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
                "Notes: This run demonstrates the feedback loop: analyst labels → labeled dataset → updated model artifact.",
            ]
        ),
        encoding="utf-8",
    )

    return metrics


def main() -> None:
    _ensure_dirs()

    # Load schema from train.csv
    train = pd.read_csv(TRAIN_PATH)
    features = _load_schema_from_train(train)

    # A) Backfill missing feature paths
    updated = backfill_feature_paths(SQLITE_PATH)
    print(f"[A] Backfilled reviews.feature_path rows: {updated}")

    # B) Export dataset
    df = export_feedback_dataset(SQLITE_PATH, features)
    print(f"[B] Exported feedback rows: {len(df)} -> {FEEDBACK_OUT}")

    # C) Retrain
    metrics = retrain_with_feedback(features)
    print(f"[C] Saved model -> {OUT_MODEL}")
    print(f"[C] Saved metrics -> {OUT_METRICS}")
    print(f"[C] Saved report -> {OUT_REPORT}")
    print("[OK] Feedback loop complete.")


if __name__ == "__main__":
    main()
