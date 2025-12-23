from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Optional, List


def init_db(db_path: Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS risk_assessments (
        id TEXT PRIMARY KEY,
        created INTEGER,
        payload_hash TEXT,
        label TEXT,
        decided_by TEXT,
        score_xgb REAL,
        ae_error REAL,
        ae_bucket TEXT,
        latency_ms REAL
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS reviews (
        id TEXT PRIMARY KEY,
        created INTEGER,
        updated INTEGER,
        risk_assessment_id TEXT,
        status TEXT,
        analyst_decision TEXT,
        notes TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS feedback_events (
        id TEXT PRIMARY KEY,
        created INTEGER,
        risk_assessment_id TEXT,
        outcome TEXT,
        notes TEXT
    )
    """)

    conn.commit()
    conn.close()


def _conn(db_path: Path):
    return sqlite3.connect(db_path)


def insert_risk_assessment(db_path: Path, rec: Dict[str, Any]):
    conn = _conn(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO risk_assessments
        (id, created, payload_hash, label, decided_by, score_xgb, ae_error, ae_bucket, latency_ms)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            rec["id"],
            rec["created"],
            rec["payload_hash"],
            rec["label"],
            rec["decided_by"],
            rec["score_xgb"],
            rec.get("ae_error"),
            rec.get("ae_bucket"),
            rec["latency_ms"],
        ),
    )
    conn.commit()
    conn.close()


def get_risk_assessment(db_path: Path, rid: str) -> Optional[Dict[str, Any]]:
    conn = _conn(db_path)
    cur = conn.cursor()
    cur.execute("SELECT * FROM risk_assessments WHERE id = ?", (rid,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    keys = ["id", "created", "payload_hash", "label", "decided_by", "score_xgb", "ae_error", "ae_bucket", "latency_ms"]
    return dict(zip(keys, row))


def create_review(db_path: Path, review: Dict[str, Any]):
    conn = _conn(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO reviews
        (id, created, updated, risk_assessment_id, status, analyst_decision, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            review["id"],
            review["created"],
            review["updated"],
            review["risk_assessment_id"],
            review["status"],
            review.get("analyst_decision"),
            review.get("notes"),
        ),
    )
    conn.commit()
    conn.close()


def list_reviews(db_path: Path, status: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = _conn(db_path)
    cur = conn.cursor()
    if status:
        cur.execute("SELECT * FROM reviews WHERE status = ? ORDER BY created DESC", (status,))
    else:
        cur.execute("SELECT * FROM reviews ORDER BY created DESC")
    rows = cur.fetchall()
    conn.close()
    keys = ["id", "created", "updated", "risk_assessment_id", "status", "analyst_decision", "notes"]
    return [dict(zip(keys, r)) for r in rows]


def get_review(db_path: Path, review_id: str) -> Optional[Dict[str, Any]]:
    conn = _conn(db_path)
    cur = conn.cursor()
    cur.execute("SELECT * FROM reviews WHERE id = ?", (review_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    keys = ["id", "created", "updated", "risk_assessment_id", "status", "analyst_decision", "notes"]
    return dict(zip(keys, row))


def update_review(db_path: Path, review_id: str, patch: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    existing = get_review(db_path, review_id)
    if not existing:
        return None

    now = int(time.time())
    status = patch.get("status", existing["status"])
    analyst_decision = patch.get("analyst_decision", existing["analyst_decision"])
    notes = patch.get("notes", existing["notes"])

    conn = _conn(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE reviews
        SET updated = ?, status = ?, analyst_decision = ?, notes = ?
        WHERE id = ?
        """,
        (now, status, analyst_decision, notes, review_id),
    )
    conn.commit()
    conn.close()
    return get_review(db_path, review_id)


def insert_feedback_event(db_path: Path, ev: Dict[str, Any]):
    conn = _conn(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO feedback_events
        (id, created, risk_assessment_id, outcome, notes)
        VALUES (?, ?, ?, ?, ?)
        """,
        (ev["id"], ev["created"], ev["risk_assessment_id"], ev["outcome"], ev.get("notes")),
    )
    conn.commit()
    conn.close()
