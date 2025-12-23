# dashboard/app.py
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import streamlit as st
import yaml

from utils import (
    append_review_log,
    find_shap_png,
    ensure_sample_shap,
    load_review_log,
    load_review_queue_jsonl,
)

CONFIG_PATH = Path("dashboard/config.yaml")


@st.cache_data(show_spinner=False)
def load_config() -> Dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def cfg_get(cfg: Dict[str, Any], *keys: str, default=None):
    cur: Any = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def sidebar(cfg: Dict[str, Any]) -> Tuple[str, float, int]:
    st.sidebar.header("Settings")

    mode_default = cfg_get(cfg, "data_source", "mode", default="api")
    mode = st.sidebar.selectbox("Data source mode", ["api", "local"], index=0 if mode_default == "api" else 1)

    st.sidebar.divider()
    st.sidebar.subheader("Queue filters")
    min_default = float(cfg_get(cfg, "queue", "min_score_xgb", default=0.0))
    min_score = st.sidebar.slider("Min XGB prob (for viewing)", 0.0, 1.0, min_default, 0.01)

    max_rows = int(cfg_get(cfg, "queue", "max_rows", default=200))
    return mode, min_score, max_rows


def set_selected_review(review_id: str):
    st.session_state["selected_review_id"] = review_id
    st.session_state["page"] = "Details"


def get_selected_review() -> Optional[str]:
    return st.session_state.get("selected_review_id")


def load_queue(cfg: Dict[str, Any], mode: str, min_score: float, max_rows: int) -> pd.DataFrame:
    """
    For this thesis-friendly setup:
    - api mode = reads artifacts/review_queue.jsonl written by API
    - local mode could be added later (but not needed right now)
    """
    if mode != "api":
        return pd.DataFrame()

    jsonl_path = cfg_get(cfg, "paths", "review_queue_jsonl", default="artifacts/review_queue.jsonl")
    items = load_review_queue_jsonl(jsonl_path, limit=max_rows)
    df = pd.DataFrame(items)

    if df.empty:
        return df

    # optional filter
    if "score_xgb" in df.columns:
        df = df[pd.to_numeric(df["score_xgb"], errors="coerce").fillna(0.0) >= float(min_score)]

    return df


def page_queue(cfg: Dict[str, Any], mode: str, min_score: float, max_rows: int):
    st.subheader("Review Queue")

    df = load_queue(cfg, mode=mode, min_score=min_score, max_rows=max_rows)
    if df.empty:
        st.info("No REVIEW events in the queue yet. Trigger some /preauth/decision calls that return REVIEW.")
        return

    # Display columns we care about
    cols = [c for c in ["id", "created", "txn_id", "timestamp", "score_xgb", "ae_error", "reason_codes", "amount", "country", "ip_country"] if c in df.columns]
    df_view = df[cols].copy()

    st.caption("Click a review_id to open Details.")
    for i, row in df_view.iterrows():
        review_id = str(row.get("id", ""))
        k = f"open_{review_id}_{i}"  # ✅ ensures unique key even if id repeats

        c1, c2, c3, c4, c5 = st.columns([2.2, 2.4, 1.2, 1.2, 4.0])

        with c1:
            if st.button(review_id, key=k):
                set_selected_review(review_id)

        with c2:
            st.write(str(row.get("timestamp", "")))

        with c3:
            st.write(row.get("score_xgb", ""))

        with c4:
            st.write(row.get("ae_error", ""))

        with c5:
            rc = row.get("reason_codes", [])
            if isinstance(rc, list) and rc:
                st.write(", ".join(map(str, rc[:6])))
            else:
                st.write("—")


def page_details(cfg: Dict[str, Any], mode: str, min_score: float, max_rows: int):
    st.subheader("Review Details")

    review_id = get_selected_review()
    if not review_id:
        st.info("Pick a review from Queue first.")
        return

    df = load_queue(cfg, mode=mode, min_score=min_score, max_rows=max_rows)
    if df.empty:
        st.warning("Queue is empty (or filters removed all rows).")
        return

    row_df = df[df["id"].astype(str) == str(review_id)] if "id" in df.columns else pd.DataFrame()
    if row_df.empty:
        st.warning("Selected review not found in current queue (maybe filters changed).")
        return

    row = row_df.iloc[0]

    # Why flagged
    st.markdown("### Why it was sent to REVIEW")
    rc = row.get("reason_codes", [])
    if isinstance(rc, list) and rc:
        for r in rc:
            st.write(f"- {r}")
    else:
        st.write("- Review due to gray-zone risk score (no reason codes attached).")

    # Scores
    st.markdown("### Scores")
    c1, c2 = st.columns(2)
    c1.metric("XGB probability", f"{row.get('score_xgb', '')}")
    c2.metric("AE error", f"{row.get('ae_error', '')}")

    # Payload snapshot
    with st.expander("Payload snapshot (thesis-safe subset)", expanded=False):
        show_cols = [c for c in ["txn_id", "timestamp", "amount", "country", "ip_country", "currency", "card_currency", "hour", "velocity_1h", "velocity_24h", "is_new_device", "is_proxy_vpn"] if c in row.index]
        st.json({c: row.get(c) for c in show_cols})

    # SHAP image by review_id
    st.markdown("### SHAP explanation")
    static_dir = cfg_get(cfg, "paths", "static_dir", default="dashboard/static")

    shap_png = find_shap_png(static_dir, str(review_id))
    fallback = ensure_sample_shap(static_dir)

    if shap_png:
        st.image(shap_png, caption=f"SHAP for review_id={review_id}", use_container_width=True)
    elif fallback:
        st.image(fallback, caption="SHAP example (no per-review SHAP found)", use_container_width=True)
        st.info(f"To show per-review SHAP, save: dashboard/static/shap_{review_id}.png")
    else:
        st.info(f"No SHAP found. Save per-review image at: dashboard/static/shap_{review_id}.png")

    # Analyst action
    st.markdown("### Analyst decision")
    analyst = st.text_input("Analyst", value=st.session_state.get("analyst", "analyst@fpn"))
    st.session_state["analyst"] = analyst

    notes = st.text_area("Notes", value="", placeholder="Add optional notes…")
    decision_time = datetime.utcnow().isoformat()

    review_log_path = cfg_get(cfg, "paths", "review_log_parquet", default="data/review_log.parquet")

    def write_decision(decision: str):
        record = {
            "review_id": str(review_id),
            "txn_id": str(row.get("txn_id", "")),
            "timestamp": str(row.get("timestamp", "")),
            "ensemble_score": None,
            "analyst_decision": decision,
            "analyst": analyst,
            "notes": notes,
            "decision_time": decision_time,
        }
        append_review_log(review_log_path, record)
        st.success(f"Saved decision: {decision}")

    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("Approve", use_container_width=True):
            write_decision("Approve")
    with b2:
        if st.button("Reject", use_container_width=True):
            write_decision("Reject")
    with b3:
        if st.button("Needs Review", use_container_width=True):
            write_decision("Needs Review")


def page_metrics(cfg: Dict[str, Any]):
    st.subheader("Metrics")

    review_log_path = cfg_get(cfg, "paths", "review_log_parquet", default="data/review_log.parquet")
    df = load_review_log(review_log_path)
    if df.empty:
        st.info("No analyst decisions yet.")
        return

    st.markdown("### Decisions")
    st.bar_chart(df["analyst_decision"].value_counts())

    if "decision_time" in df.columns:
        st.markdown("### Latest decisions")
        st.dataframe(df.sort_values("decision_time", ascending=False).head(30), use_container_width=True)


def main():
    st.set_page_config(page_title="FPN Review Dashboard", layout="wide")
    cfg = load_config()

    st.title(cfg_get(cfg, "app", "title", default="FPN Review Dashboard"))

    mode, min_score, max_rows = sidebar(cfg)

    page = st.sidebar.radio("Pages", ["Queue", "Details", "Metrics"], index=["Queue", "Details", "Metrics"].index(st.session_state.get("page", "Queue")))
    st.session_state["page"] = page

    if page == "Queue":
        page_queue(cfg, mode=mode, min_score=min_score, max_rows=max_rows)
    elif page == "Details":
        page_details(cfg, mode=mode, min_score=min_score, max_rows=max_rows)
    else:
        page_metrics(cfg)


if __name__ == "__main__":
    main()
