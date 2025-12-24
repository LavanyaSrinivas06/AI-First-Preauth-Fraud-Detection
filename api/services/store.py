# api/services/store.py
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

def _ensure_column(con: sqlite3.Connection, table: str, col: str, col_type: str) -> None:
    """
    Lightweight migration: add column if missing.
    Safe for SQLite when running locally.
    """
    cur = con.cursor()
    cur.execute(f"PRAGMA table_info({table});")
    cols = {row[1] for row in cur.fetchall()}  # row[1] = name
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

              status TEXT DEFAULT 'open', -- open | closed
              analyst_decision TEXT,      -- APPROVE | BLOCK
              analyst TEXT,
              notes TEXT,
              updated INTEGER
            )
            """
        )

        # Decision log table (stores APPROVE/REVIEW/BLOCK for audit + thesis)
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
              payload_min TEXT            -- JSON string

              txn_id TEXT,
              timestamp TEXT
            )
            """
        )

        # Feedback events (future loop)
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

            # ---- migrations (for existing DBs) ----
        _ensure_column(con, "reviews", "txn_id", "TEXT")
        _ensure_column(con, "reviews", "timestamp", "TEXT")
        _ensure_column(con, "decisions", "txn_id", "TEXT")
        _ensure_column(con, "decisions", "timestamp", "TEXT")

        con.commit()
    finally:
        con.close()

def _is_checkout_payload(payload: Dict[str, Any]) -> bool:
    # A lightweight heuristic: if these exist, it's checkout-like
    keys = {"amount", "country", "ip_country", "currency", "card_currency"}
    return any(k in payload for k in keys)


def _is_processed_102(payload: Dict[str, Any]) -> bool:
    # Another heuristic: processed features typically have num__/cat__ prefixes
    if any(k.startswith("num__") for k in payload.keys()):
        return True
    if any(k.startswith("cat__") for k in payload.keys()):
        return True
    return False

def _payload_min_checkout(payload: Dict[str, Any]) -> Dict[str, Any]:
    # thesis-safe subset for checkout-like payload
    return {
        "txn_id": payload.get("txn_id"),
        "timestamp": payload.get("timestamp"),
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

def _payload_min_processed(payload: Dict[str, Any], top_v: int = 8, top_cat: int = 25) -> Dict[str, Any]:
    """
    For processed-102 payloads:
    - keep key engineered numeric features (if present)
    - keep top |V| numeric components (V1..V28 etc.)
    - keep cat__ flags that are 1/True (limited)
    """
    out: Dict[str, Any] = {"_schema": "processed_102_min"}

    # always include common, human-friendly engineered features if they exist
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

    # pick top |V| from any numeric keys that look like "num__V*"
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

    # include cat__ flags that are active (1/True)
    cats: List[str] = []
    for k, v in payload.items():
        if not isinstance(k, str) or not k.startswith("cat__"):
            continue
        # accept 1 / True / 1.0
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

def _payload_min(payload: Dict[str, Any]) -> Dict[str, Any]:
    # If payload is processed-102, store processed snapshot.
    # If payload is checkout-like, store checkout snapshot.
    if _is_processed_102(payload) and not _is_checkout_payload(payload):
        return _payload_min_processed(payload)
    return _payload_min_checkout(payload)

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
) -> str:
    """
    Logs EVERY decision (APPROVE/REVIEW/BLOCK). Lightweight but very useful for thesis + dashboard metrics.
    """
    init_db(sqlite_path)
    now = int(time.time())
    dec_id = f"dec_{payload_hash[:16]}_{now}"

    con = sqlite3.connect(str(sqlite_path))
    try:
        cur = con.cursor()
        cur.execute(
            """
            INSERT INTO decisions (
              id, created, decision, payload_hash,
              score_xgb, ae_error, ae_percentile_vs_legit, ae_bucket,
              reason_codes, payload_min,
                txn_id, timestamp
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                json.dumps(_payload_min(payload)),
                meta.get("txn_id"),
                meta.get("timestamp"),
                
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
) -> Optional[str]:
    """
    Persist only REVIEW events (queue items shown on dashboard).
    Returns review_id if saved, else None.
    """
    if decision != "REVIEW":
        return None

    init_db(sqlite_path)
    now = int(time.time())
    review_id = f"rev_{payload_hash[:16]}"

    con = sqlite3.connect(str(sqlite_path))
    try:
        cur = con.cursor()

        # idempotent
        cur.execute("SELECT id FROM reviews WHERE id = ?", (review_id,))
        if cur.fetchone():
            return review_id

        cur.execute(
            """
            INSERT INTO reviews (
              id, created, txn_id, timestamp, decision,
              score_xgb, ae_error, ae_percentile_vs_legit, ae_bucket,
              payload_hash, reason_codes, payload_min,
              status, analyst_decision, analyst, notes, updated
            )
            VALUES (?, ?, ?, ?, 'REVIEW', ?, ?, ?, ?, ?, ?, ?, 'open', NULL, NULL, NULL, ?)
            """,
            (
                review_id,
                now,
                meta.get("txn_id"),
                meta.get("timestamp"),
                float(p_xgb) if p_xgb is not None else None,
                float(ae_err) if ae_err is not None else None,
                float(ae_percentile) if ae_percentile is not None else None,
                ae_bucket,
                payload_hash,
                json.dumps(reason_codes),
                json.dumps(_payload_min(payload)),
                now,
            ),
        )
        con.commit()
        return review_id
    finally:
        con.close()


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
            d["reason_codes"] = json.loads(d["reason_codes"] or "[]")
            d["payload_min"] = json.loads(d["payload_min"] or "{}")
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
        d["reason_codes"] = json.loads(d["reason_codes"] or "[]")
        d["payload_min"] = json.loads(d["payload_min"] or "{}")
        return d
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


def insert_feedback_event(
    sqlite_path: Path,
    review_id: str,
    outcome: str,
    notes: Optional[str] = None,
) -> str:
    init_db(sqlite_path)
    now = int(time.time())
    fb_id = f"fb_{int(time.time())}_{review_id[-6:]}"

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
