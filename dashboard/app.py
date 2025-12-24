# dashboard/app.py
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import streamlit as st
import yaml

# ✅ API helpers (you should have these in dashboard/utils_api.py)
# - api_get_review_queue() -> List[Dict]
# - api_get_review(review_id: str) -> Dict
# - api_close_review(review_id: str, analyst: str, decision: str, notes: str|None) -> Dict
from utils_api import api_get_review_queue, api_get_review, api_close_review


from utils import (
    find_shap_png,
    ensure_sample_shap,
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


def load_queue_from_api(min_score: float, max_rows: int) -> pd.DataFrame:
    items = api_get_review_queue() or []
    if max_rows:
        items = items[: int(max_rows)]
    df = pd.DataFrame(items)
    if df.empty:
        return df

    if "score_xgb" in df.columns:
        df["score_xgb"] = pd.to_numeric(df["score_xgb"], errors="coerce")
        df = df[df["score_xgb"].fillna(0.0) >= float(min_score)]

    return df


def page_queue(cfg: Dict[str, Any], mode: str, min_score: float, max_rows: int):
    st.subheader("Review Queue")

    if mode != "api":
        st.info("Local mode not enabled in this version. Switch Data source mode to API.")
        return

    df = load_queue_from_api(min_score=min_score, max_rows=max_rows)
    if df.empty:
        st.info("No REVIEW events in the queue yet. Trigger /preauth/decision that returns REVIEW.")
        return

    cols = [c for c in ["id", "created", "txn_id", "timestamp", "score_xgb", "ae_error", "ae_bucket", "ae_percentile_vs_legit", "reason_codes"] if c in df.columns]
    df_view = df[cols].copy()

    st.caption("Click a review_id to open Details.")
    for i, row in df_view.iterrows():
        review_id = str(row.get("id", ""))
        k = f"open_{review_id}_{i}"  # unique key

        c1, c2, c3, c4, c5 = st.columns([2.2, 2.4, 1.2, 1.2, 4.0])

        with c1:
            if st.button(review_id, key=k):
                set_selected_review(review_id)

        with c2:
            st.write(str(row.get("timestamp", "")))

        with c3:
            st.write(row.get("score_xgb", ""))

        with c4:
            # show readable AE if present
            bkt = row.get("ae_bucket", "")
            pct = row.get("ae_percentile_vs_legit", "")
            ae_txt = "—"
            if pd.notna(pct) and pct != "":
                ae_txt = f"{bkt} ({float(pct):.2f}pctl)"
            st.write(ae_txt)

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

    if mode != "api":
        st.info("Local mode not enabled in this version. Switch Data source mode to API.")
        return

    # ✅ fetch detail from API (source of truth)
    try:
        review = api_get_review(review_id)
    except Exception as e:
        st.error(f"Failed to fetch review: {type(e).__name__}: {e}")
        return

    if not isinstance(review, dict) or not review:
        st.warning("Review not found.")
        return

    # Why flagged
    st.markdown("### Why it was sent to REVIEW")
    rc = review.get("reason_codes", [])
    if isinstance(rc, list) and rc:
        for r in rc:
            st.write(f"- {r}")
    else:
        st.write("- Review due to gray-zone score.")

    # Scores
    st.markdown("### Scores")
    c1, c2, c3 = st.columns(3)
    c1.metric("XGB probability", f"{review.get('score_xgb', '')}")
    c2.metric("AE bucket", f"{review.get('ae_bucket', '')}")
    pct = review.get("ae_percentile_vs_legit", None)
    c3.metric("AE percentile vs legit", (f"{float(pct):.2f}" if pct is not None else "—"))

    # Payload snapshot (thesis-safe)
   # Payload snapshot
    with st.expander("Payload snapshot (stored subset)", expanded=False):
        # DB/API returns payload_min as dict (your top features / minimal payload)
        pm = review.get("payload_min", None)

        if isinstance(pm, dict) and pm:
            st.json(pm)
        else:
            # fallback if payload_min wasn't present
            st.info("No payload_min found for this review.")

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

    # ✅ Analyst action -> close review via API (no parquet/local logs)
    st.markdown("### Analyst decision (closes ticket)")
    analyst = st.text_input("Analyst", value=st.session_state.get("analyst", "analyst@fpn"))
    st.session_state["analyst"] = analyst
    notes = st.text_area("Notes", value="", placeholder="Optional notes…")

    def submit_decision(decision: str):
        try:
            api_close_review(
                review_id=review_id,
                analyst=analyst,
                decision=decision,  # "APPROVE" or "BLOCK"
                notes=notes,
            )
            st.success(f"Review {review_id} closed as {decision}.")
            st.session_state["page"] = "Queue"
            # refresh cached queue
            st.cache_data.clear()
        except Exception as e:
            st.error(f"Failed to close review: {type(e).__name__}: {e}")

    b1, b2 = st.columns(2)
    with b1:
        if st.button("Approve", use_container_width=True):
            submit_decision("APPROVE")
    with b2:
        if st.button("Block", use_container_width=True):
            submit_decision("BLOCK")


def page_metrics(cfg: Dict[str, Any]):
    st.subheader("Metrics")
    st.info("Metrics page will be added after close-review flow is stable (DB-driven).")


def main():
    st.set_page_config(page_title="FPN Review Dashboard", layout="wide")
    cfg = load_config()

    st.title(cfg_get(cfg, "app", "title", default="FPN Review Dashboard"))

    mode, min_score, max_rows = sidebar(cfg)

    page = st.sidebar.radio(
        "Pages",
        ["Queue", "Details", "Metrics"],
        index=["Queue", "Details", "Metrics"].index(st.session_state.get("page", "Queue")),
    )
    st.session_state["page"] = page

    if page == "Queue":
        page_queue(cfg, mode=mode, min_score=min_score, max_rows=max_rows)
    elif page == "Details":
        page_details(cfg, mode=mode, min_score=min_score, max_rows=max_rows)
    else:
        page_metrics(cfg)


if __name__ == "__main__":
    main()
