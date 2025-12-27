# api/services/store.py
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


# -------------------------
# Feature snapshot (for real per-ticket SHAP)
# -------------------------
def save_feature_snapshot(artifacts_dir: Path, review_id: str, features: Dict[str, Any]) -> str:
    """
    Persist the exact processed-102 features used for scoring.
    Returns a relative path string to store in DB (portable).
    """
    snap_dir = artifacts_dir / "feature_snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)

    snap_path = snap_dir / f"{review_id}.json"
    snap_path.write_text(json.dumps(features, ensure_ascii=False), encoding="utf-8")

    # store relative path (from repo root perspective)
    return str(snap_path.as_posix())


# -------------------------
# DB initialization + migration
# -------------------------
def _ensure_column(con: sqlite3.Connection, table: str, col: str, col_type: str) -> None:
    """Lightweight migration: add column if missing (SQLite-safe for local use)."""
    cur = con.cursor()
    cur.execute(f"PRAGMA table_info({table});")
    cols = {row[1] for row in cur.fetchall()}
    if col not in cols:
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type};")


def init_db(sqlite_path: Path) -> None:
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(str(sqlite_path))
    try:
        cur = con.cursor()

        # REVIEW queue table
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS reviews (
              id TEXT PRIMARY KEY,
              created INTEGER NOT NULL,
              txn_id TEXT,
              timestamp TEXT,
              decision TEXT NOT NULL,     -- always "REVIEW"
              score_xgb REAL,
              ae_error REAL,
              ae_percentile_vs_legit REAL,
              ae_bucket TEXT,
              payload_hash TEXT,
              reason_codes TEXT,          -- JSON string
              payload_min TEXT,           -- JSON string
              feature_path TEXT,          -- optional: path to saved feature snapshot (.json)

              status TEXT DEFAULT 'open', -- open | closed
              analyst_decision TEXT,      -- APPROVE | BLOCK
              analyst TEXT,
              notes TEXT,
              updated INTEGER
            )
            """
        )

        # Decision log table (audit)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS decisions (
              id TEXT PRIMARY KEY,
              created INTEGER NOT NULL,
              decision TEXT NOT NULL,     -- APPROVE | REVIEW | BLOCK
              payload_hash TEXT NOT NULL,
              score_xgb REAL,
              ae_error REAL,
              ae_percentile_vs_legit REAL,
              ae_bucket TEXT,
              reason_codes TEXT,          -- JSON string
              payload_min TEXT,           -- JSON string
              feature_path TEXT,          -- optional: path to saved feature snapshot (.json)
              txn_id TEXT,
              timestamp TEXT
            )
            """
        )

        # Feedback events
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback_events (
              id TEXT PRIMARY KEY,
              created INTEGER NOT NULL,
              review_id TEXT NOT NULL,
              outcome TEXT NOT NULL,
              notes TEXT
            )
            """
        )

        # ---- migrations (for older DBs) ----
        _ensure_column(con, "reviews", "txn_id", "TEXT")
        _ensure_column(con, "reviews", "timestamp", "TEXT")
        _ensure_column(con, "reviews", "feature_path", "TEXT")
        _ensure_column(con, "decisions", "txn_id", "TEXT")
        _ensure_column(con, "decisions", "timestamp", "TEXT")
        _ensure_column(con, "decisions", "feature_path", "TEXT")

        con.commit()
    finally:
        con.close()


# -------------------------
# Payload snapshot helpers
# -------------------------
def _get_txn_id(payload: Dict[str, Any], meta: Optional[Dict[str, Any]] = None) -> Optional[str]:
    meta = meta or {}
    return (
        meta.get("txn_id")
        or meta.get("transaction_id")
        or payload.get("txn_id")
        or payload.get("transaction_id")
        or payload.get("id")
    )


def _get_timestamp(payload: Dict[str, Any], meta: Optional[Dict[str, Any]] = None) -> Optional[str]:
    meta = meta or {}
    return (
        meta.get("timestamp")
        or payload.get("timestamp")
        or payload.get("created_at")
        or payload.get("created")
        or payload.get("time")
    )


def _is_checkout_payload(payload: Dict[str, Any]) -> bool:
    keys = {"amount", "country", "ip_country", "currency", "card_currency"}
    return any(k in payload for k in keys)


def _is_processed_102(payload: Dict[str, Any]) -> bool:
    return any(isinstance(k, str) and (k.startswith("num__") or k.startswith("cat__")) 
               for k in payload.keys())


def _payload_min_checkout(payload: Dict[str, Any], meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    txn_id = _get_txn_id(payload, meta)
    ts = _get_timestamp(payload, meta)

    return {
        "txn_id": txn_id,
        "transaction_id": txn_id,  # alias for UI consistency
        "timestamp": ts,
        "created_at": ts,          # alias
        "amount": payload.get("amount"),
        "country": payload.get("country"),
        "ip_country": payload.get("ip_country"),
        "currency": payload.get("currency"),
        "card_currency": payload.get("card_currency"),
        "hour": payload.get("hour"),
        "velocity_1h": payload.get("velocity_1h"),
        "velocity_24h": payload.get("velocity_24h"),
        "is_new_device": payload.get("is_new_device"),
        "is_proxy_vpn": payload.get("is_proxy_vpn"),
        "_schema": "checkout_min",
    }


def _payload_min_processed(
    payload: Dict[str, Any],
    meta: Optional[Dict[str, Any]] = None,
    top_v: int = 8,
    top_cat: int = 25,
) -> Dict[str, Any]:
    txn_id = _get_txn_id(payload, meta)
    ts = _get_timestamp(payload, meta)

    out: Dict[str, Any] = {
        "_schema": "processed_102_min_v1",
        "txn_id": txn_id,
        "transaction_id": txn_id,
        "timestamp": ts,
        "created_at": ts,
    }

    keep_num = [
        "num__Amount",
        "num__ip_reputation",
        "num__txn_count_5m",
        "num__txn_count_30m",
        "num__txn_count_60m",
        "num__geo_distance_km",
        "num__account_age_days",
        "num__token_age_days",
        "num__amount_zscore",
        "num__avg_amount_7d",
        "num__avg_spend_user_30d",
    ]
    for k in keep_num:
        if k in payload:
            out[k] = payload.get(k)

    # top |V| components
    v_vals: List[tuple[str, float]] = []
    for k, v in payload.items():
        if not isinstance(k, str) or not k.startswith("num__V"):
            continue
        try:
            fv = float(v)
        except Exception:
            continue
        v_vals.append((k, fv))

    v_vals.sort(key=lambda kv: abs(kv[1]), reverse=True)
    for k, fv in v_vals[: max(0, int(top_v))]:
        out[k] = fv

    # cat flags on
    cats: List[str] = []
    for k, v in payload.items():
        if not isinstance(k, str) or not k.startswith("cat__"):
            continue

        is_on = False
        if isinstance(v, bool) and v:
            is_on = True
        elif isinstance(v, (int, float)) and float(v) == 1.0:
            is_on = True

        if is_on:
            cats.append(k)

    cats = cats[: max(0, int(top_cat))]
    if cats:
        out["cat_flags_on"] = cats

    return out


def _payload_min(payload: Dict[str, Any], meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if _is_processed_102(payload) and not _is_checkout_payload(payload):
        return _payload_min_processed(payload, meta=meta)
    return _payload_min_checkout(payload, meta=meta)


# -------------------------
# Core persistence
# -------------------------
def log_decision(
    sqlite_path: Path,
    decision: str,
    payload: Dict[str, Any],
    meta: Dict[str, Any],
    p_xgb: Optional[float],
    ae_err: Optional[float],
    payload_hash: str,
    reason_codes: List[str],
    ae_percentile: Optional[float] = None,
    ae_bucket: Optional[str] = None,
    feature_path: Optional[str] = None,
) -> str:
    """Logs EVERY decision (APPROVE/REVIEW/BLOCK)."""
    init_db(sqlite_path)
    now = int(time.time())
    dec_id = f"dec_{payload_hash[:16]}_{now}"

    txn_id = _get_txn_id(payload, meta)
    ts = _get_timestamp(payload, meta)

    con = sqlite3.connect(str(sqlite_path))
    try:
        cur = con.cursor()
        cur.execute(
            """
            INSERT INTO decisions (
              id, created, decision, payload_hash,
              score_xgb, ae_error, ae_percentile_vs_legit, ae_bucket,
              reason_codes, payload_min, feature_path,
              txn_id, timestamp
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                dec_id,
                now,
                decision,
                payload_hash,
                float(p_xgb) if p_xgb is not None else None,
                float(ae_err) if ae_err is not None else None,
                float(ae_percentile) if ae_percentile is not None else None,
                ae_bucket,
                json.dumps(reason_codes),
                json.dumps(_payload_min(payload, meta)),
                feature_path,
                txn_id,
                ts,
            ),
        )
        con.commit()
        return dec_id
    finally:
        con.close()


def save_review_if_needed(
    sqlite_path: Path,
    decision: str,
    payload: Dict[str, Any],
    meta: Dict[str, Any],
    p_xgb: Optional[float],
    ae_err: Optional[float],
    payload_hash: str,
    reason_codes: List[str],
    ae_percentile: Optional[float] = None,
    ae_bucket: Optional[str] = None,
    feature_path: Optional[str] = None,   # ✅ ADD THIS
) -> Optional[str]:
    """
    Persist only REVIEW events (queue items shown on dashboard).

    IMPORTANT (Option B SHAP):
    - If artifacts_dir is provided, we save a feature snapshot JSON and store feature_path.
    """
    if decision != "REVIEW":
        return None

    init_db(sqlite_path)
    now = int(time.time())
    review_id = f"rev_{payload_hash[:16]}"

    txn_id = _get_txn_id(payload, meta)
    ts = _get_timestamp(payload, meta)

    # If we can, persist the exact feature vector used for scoring
    con = sqlite3.connect(str(sqlite_path))
    try:
        cur = con.cursor()
        # idempotent
        cur.execute("SELECT id FROM reviews WHERE id = ?", (review_id,))
        if cur.fetchone():
            # if existing and feature_path is missing, attempt to backfill
            if feature_path:
                cur.execute("UPDATE reviews SET feature_path=? WHERE id=? AND (feature_path IS NULL OR feature_path='')", (feature_path, review_id))
                con.commit()
            return review_id

        cur.execute(
            """
            INSERT INTO reviews (
                id, created, txn_id, timestamp, decision,
                score_xgb, ae_error, ae_percentile_vs_legit, ae_bucket,
                payload_hash, reason_codes, payload_min, feature_path,
                status, analyst_decision, analyst, notes, updated
            )
            VALUES (?, ?, ?, ?, 'REVIEW', ?, ?, ?, ?, ?, ?, ?, ?, 'open', NULL, NULL, NULL, ?)
            """,
            (
                review_id,
                now,
                 txn_id,
                ts,
                float(p_xgb) if p_xgb is not None else None,
                float(ae_err) if ae_err is not None else None,
                float(ae_percentile) if ae_percentile is not None else None,
                ae_bucket,
                payload_hash,
                json.dumps(reason_codes),
                json.dumps(_payload_min(payload, meta)),
                feature_path,   # ✅ HERE
                 now,
            ),
        )
        con.commit()
        return review_id
    finally:
        con.close()


#--------------------------
# Review queue + retrieval
# -------------------------

def load_review_queue(sqlite_path: Path, limit: int = 200) -> List[Dict[str, Any]]:
    init_db(sqlite_path)

    con = sqlite3.connect(str(sqlite_path))
    con.row_factory = sqlite3.Row
    try:
        cur = con.cursor()
        cur.execute(
            """
            SELECT *
            FROM reviews
            WHERE status = 'open'
            ORDER BY created DESC
            LIMIT ?
            """,
            (int(limit),),
        )
        rows = cur.fetchall()

        out: List[Dict[str, Any]] = []
        for r in rows:
            d = dict(r)
            d["reason_codes"] = json.loads(d.get("reason_codes") or "[]")
            d["payload_min"] = json.loads(d.get("payload_min") or "{}")
            out.append(d)
        return out
    finally:
        con.close()


def get_review_by_id(sqlite_path: Path, review_id: str) -> Optional[Dict[str, Any]]:
    init_db(sqlite_path)

    con = sqlite3.connect(str(sqlite_path))
    con.row_factory = sqlite3.Row
    try:
        cur = con.cursor()
        cur.execute("SELECT * FROM reviews WHERE id = ?", (review_id,))
        row = cur.fetchone()
        if not row:
            return None
        d = dict(row)
        d["reason_codes"] = json.loads(d.get("reason_codes") or "[]")
        d["payload_min"] = json.loads(d.get("payload_min") or "{}")
        return d
    finally:
        con.close()

def _update_review_feature_path_if_missing(sqlite_path: Path, review_id: str, feature_path: str) -> None:
    """
    If the review row exists already and feature_path is NULL/empty, fill it.
    """
    if not feature_path:
        return

    con = sqlite3.connect(str(sqlite_path))
    try:
        cur = con.cursor()
        cur.execute(
            """
            UPDATE reviews
            SET feature_path = ?
            WHERE id = ?
              AND (feature_path IS NULL OR feature_path = '')
            """,
            (feature_path, review_id),
        )
        con.commit()
    finally:
        con.close()


def update_review(
    sqlite_path: Path,
    review_id: str,
    analyst_decision: str,
    analyst: str,
    notes: Optional[str] = None,
) -> bool:
    init_db(sqlite_path)
    now = int(time.time())

    con = sqlite3.connect(str(sqlite_path))
    try:
        cur = con.cursor()
        cur.execute(
            """
            UPDATE reviews
            SET status='closed',
                analyst_decision=?,
                analyst=?,
                notes=?,
                updated=?
            WHERE id=?
            """,
            (analyst_decision, analyst, notes, now, review_id),
        )
        con.commit()
        return cur.rowcount > 0
    finally:
        con.close()


# -------------------------
# Feedback events + export
# -------------------------
def insert_feedback_event(
    sqlite_path: Path,
    review_id: str,
    outcome: str,
    notes: Optional[str] = None,
) -> str:
    init_db(sqlite_path)
    now = int(time.time())
    fb_id = f"fb_{now}_{review_id[-6:]}"

    con = sqlite3.connect(str(sqlite_path))
    try:
        cur = con.cursor()
        cur.execute(
            """
            INSERT INTO feedback_events (id, created, review_id, outcome, notes)
            VALUES (?, ?, ?, ?, ?)
            """,
            (fb_id, now, review_id, outcome, notes),
        )
        con.commit()
        return fb_id
    finally:
        con.close()


def list_feedback_events(sqlite_path: Path, limit: int = 200) -> List[Dict[str, Any]]:
    init_db(sqlite_path)

    con = sqlite3.connect(str(sqlite_path))
    con.row_factory = sqlite3.Row
    try:
        cur = con.cursor()
        cur.execute(
            """
            SELECT *
            FROM feedback_events
            ORDER BY created DESC
            LIMIT ?
            """,
            (int(limit),),
        )
        return [dict(r) for r in cur.fetchall()]
    finally:
        con.close()


def feedback_summary(sqlite_path: Path) -> Dict[str, Any]:
    init_db(sqlite_path)

    con = sqlite3.connect(str(sqlite_path))
    con.row_factory = sqlite3.Row
    try:
        cur = con.cursor()

        cur.execute("SELECT COUNT(*) AS n FROM feedback_events")
        n_total = int(cur.fetchone()["n"])

        cur.execute(
            """
            SELECT outcome, COUNT(*) AS n
            FROM feedback_events
            GROUP BY outcome
            """
        )
        by_outcome = {row["outcome"]: int(row["n"]) for row in cur.fetchall()}

        cur.execute("SELECT COUNT(*) AS n FROM reviews WHERE status='closed'")
        n_closed = int(cur.fetchone()["n"])

        return {
            "feedback_total": n_total,
            "feedback_by_outcome": by_outcome,
            "reviews_closed_total": n_closed,
        }
    finally:
        con.close()


def export_feedback_samples(sqlite_path: Path, limit: int = 1000) -> List[Dict[str, Any]]:
    """
    Export labeled samples for retraining/analysis.
    Uses CLOSED reviews as labels:
      BLOCK => label 1 (fraud)
      APPROVE => label 0 (legit)
    """
    init_db(sqlite_path)

    con = sqlite3.connect(str(sqlite_path))
    con.row_factory = sqlite3.Row
    try:
        cur = con.cursor()
        cur.execute(
            """
            SELECT *
            FROM reviews
            WHERE status='closed' AND analyst_decision IS NOT NULL
            ORDER BY updated DESC
            LIMIT ?
            """,
            (int(limit),),
        )

        rows = cur.fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            d = dict(r)
            d["reason_codes"] = json.loads(d.get("reason_codes") or "[]")
            d["payload_min"] = json.loads(d.get("payload_min") or "{}")

            if d.get("analyst_decision") == "BLOCK":
                d["label"] = 1
            elif d.get("analyst_decision") == "APPROVE":
                d["label"] = 0
            else:
                d["label"] = None

            out.append(d)

        return out
    finally:
        con.close()
