from __future__ import annotations

import sqlite3
from pathlib import Path

from api.services import store


def test_init_db_creates_tables(init_test_db: Path):
    con = sqlite3.connect(str(init_test_db))
    cur = con.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
    tables = [r[0] for r in cur.fetchall()]
    con.close()

    assert "decisions" in tables
    assert "reviews" in tables
    assert "feedback_events" in tables


def test_migration_adds_model_version_columns(init_test_db: Path):
    con = sqlite3.connect(str(init_test_db))
    cur = con.cursor()

    cur.execute("PRAGMA table_info(decisions);")
    cols_dec = {r[1] for r in cur.fetchall()}

    cur.execute("PRAGMA table_info(reviews);")
    cols_rev = {r[1] for r in cur.fetchall()}
    con.close()

    assert "model_version" in cols_dec
    assert "model_version" in cols_rev


def test_log_decision_inserts_model_version(init_test_db: Path):
    dec_id = store.log_decision(
        sqlite_path=init_test_db,
        decision="BLOCK",
        payload={"num__V1": 0.0, "cat__x": 1},  # minimal payload ok for store
        meta={},
        p_xgb=0.9,
        ae_err=None,
        payload_hash="abc123",
        reason_codes=["xgb_high_risk"],
        model_version="xgb-feedback-2026w01",
    )

    con = sqlite3.connect(str(init_test_db))
    cur = con.cursor()
    cur.execute("SELECT id, model_version FROM decisions WHERE id = ?", (dec_id,))
    row = cur.fetchone()
    con.close()

    assert row is not None
    assert row[1] == "xgb-feedback-2026w01"
