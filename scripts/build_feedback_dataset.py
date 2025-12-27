# scripts/build_feedback_dataset.py
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import pandas as pd

SQLITE_PATH = Path("artifacts/inference_store.sqlite")
OUT_PATH = Path("data/processed/feedback_labeled.csv")


def load_closed_reviews(sqlite_path: Path) -> pd.DataFrame:
    con = sqlite3.connect(str(sqlite_path))
    con.row_factory = sqlite3.Row
    try:
        df = pd.read_sql_query(
            """
            SELECT id, analyst_decision, analyst, notes, updated, feature_path
            FROM reviews
            WHERE status='closed' AND analyst_decision IS NOT NULL
            ORDER BY updated DESC
            """,
            con,
        )
        return df
    finally:
        con.close()


def load_feature_snapshot(p: str) -> dict:
    # feature_path may be relative or absolute
    fp = Path(p)
    if not fp.exists():
        # try relative to repo root
        fp = Path(".") / p
    if not fp.exists():
        raise FileNotFoundError(f"feature snapshot not found: {p}")
    return json.loads(fp.read_text(encoding="utf-8"))


def main():
    df = load_closed_reviews(SQLITE_PATH)
    if df.empty:
        print("No closed reviews found. Close at least 1 review first.")
        return

    rows = []
    for _, r in df.iterrows():
        decision = str(r["analyst_decision"]).upper().strip()
        if decision not in {"APPROVE", "BLOCK"}:
            continue

        label = 1 if decision == "BLOCK" else 0
        feature_path = r.get("feature_path")

        if not feature_path or str(feature_path).strip() == "":
            # if feature_path missing, skip (we need full 102 features for retraining)
            continue

        feats = load_feature_snapshot(str(feature_path))

        # build one row: 102 features + metadata + label
        row = dict(feats)
        row["label"] = label
        row["review_id"] = r["id"]
        row["analyst"] = r.get("analyst")
        row["notes"] = r.get("notes")
        row["updated"] = r.get("updated")
        rows.append(row)

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        print("No rows exported. Likely feature_path missing on closed reviews.")
        return

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(out_df)} labeled feedback rows -> {OUT_PATH}")


if __name__ == "__main__":
    main()
