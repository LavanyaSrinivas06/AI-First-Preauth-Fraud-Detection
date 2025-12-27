# dashboard/app.py
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import streamlit as st
import yaml

from utils_api import (
    api_get_review_queue,
    api_get_review,
    api_close_review,
    api_get_feedback_export,  # closed tickets export
)

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


def clear_selected_review():
    st.session_state.pop("selected_review_id", None)


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


def _fmt_dt(v: Any) -> str:
    if v is None or v == "":
        return "—"
    return str(v)


def _fmt_reason_codes(rc: Any) -> str:
    if isinstance(rc, list) and rc:
        return ", ".join(map(str, rc[:6]))
    if isinstance(rc, str) and rc.strip():
        return rc
    return "—"


def render_details(cfg: Dict[str, Any], mode: str, review_id: str):
    st.markdown("---")
    st.subheader(f"Details: {review_id}")

    if mode != "api":
        st.info("Local mode not enabled in this version. Switch Data source mode to API.")
        return

    try:
        review = api_get_review(review_id)
    except Exception as e:
        st.error(f"Failed to fetch review: {type(e).__name__}: {e}")
        return

    if not isinstance(review, dict) or not review:
        st.warning("Review not found.")
        return

    # Why
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

    # Payload snapshot
    with st.expander("Payload snapshot (stored subset)", expanded=False):
        pm = review.get("payload_min", None)
        if isinstance(pm, dict) and pm:
            st.json(pm)
        else:
            st.info("No payload_min found for this review.")

    # SHAP
    st.markdown("### SHAP explanation")
    static_dir = cfg_get(cfg, "paths", "static_dir", default="dashboard/static")
    shap_png = find_shap_png(static_dir, str(review_id))
    fallback = ensure_sample_shap(static_dir)

    if shap_png:
        st.image(shap_png, caption=f"SHAP for review_id={review_id}", use_container_width=True)
    else:
        st.warning("Per-review SHAP not found. You are seeing a fallback placeholder.")
        if fallback:
            st.image(fallback, caption="Fallback SHAP example", use_container_width=True)
        st.info(f"Expected file: dashboard/static/shap_{review_id}.png")

    # Close review
    st.markdown("### Analyst decision (closes ticket)")
    analyst = st.text_input("Analyst", value=st.session_state.get("analyst", "analyst@fpn"))
    st.session_state["analyst"] = analyst
    notes = st.text_area("Notes", value="", placeholder="Optional notes…")

    def submit_decision(decision: str):
        try:
            api_close_review(
                review_id=review_id,
                analyst=analyst,
                decision=decision,
                notes=notes,
            )
            st.success(f"Review {review_id} closed as {decision}.")
            clear_selected_review()
            st.cache_data.clear()
        except Exception as e:
            st.error(f"Failed to close review: {type(e).__name__}: {e}")

    b1, b2, b3 = st.columns([1, 1, 1])
    with b1:
        if st.button("Approve", use_container_width=True):
            submit_decision("APPROVE")
    with b2:
        if st.button("Block", use_container_width=True):
            submit_decision("BLOCK")
    with b3:
        if st.button("Back", use_container_width=True):
            clear_selected_review()


def page_queue(cfg: Dict[str, Any], mode: str, min_score: float, max_rows: int):
    st.subheader("Review Queue (Open)")

    if mode != "api":
        st.info("Local mode not enabled in this version. Switch Data source mode to API.")
        return

    df = load_queue_from_api(min_score=min_score, max_rows=max_rows)
    if df.empty:
        st.info("No REVIEW events in the queue yet. Trigger /preauth/decision that returns REVIEW.")
        return

    if "timestamp" not in df.columns:
        df["timestamp"] = None
    if "reason_codes" not in df.columns:
        df["reason_codes"] = None

    st.caption("Shows open REVIEW tickets. Click View to see details inline (no sidebar Details page).")

    h1, h2, h3, h4 = st.columns([2.2, 2.6, 4.8, 1.2])
    with h1:
        st.markdown("**Review ID**")
    with h2:
        st.markdown("**Time**")
    with h3:
        st.markdown("**Why review**")
    with h4:
        st.markdown("**Action**")

    st.divider()

    for i, row in df.iterrows():
        review_id = str(row.get("id", ""))
        ts = _fmt_dt(row.get("timestamp", None))
        why = _fmt_reason_codes(row.get("reason_codes", None))

        c1, c2, c3, c4 = st.columns([2.2, 2.6, 4.8, 1.2])
        with c1:
            st.write(review_id)
        with c2:
            st.write(ts)
        with c3:
            st.write(why)
        with c4:
            if st.button("View", key=f"view_{review_id}_{i}", use_container_width=True):
                set_selected_review(review_id)

    sel = get_selected_review()
    if sel:
        render_details(cfg, mode=mode, review_id=sel)


def page_log(cfg: Dict[str, Any], mode: str):
    st.subheader("Closed Tickets Log (APPROVE / BLOCK)")

    if mode != "api":
        st.info("Local mode not enabled in this version. Switch Data source mode to API.")
        return

    top = st.columns([1.2, 1.6, 2.8, 1.0])
    with top[0]:
        limit = st.number_input("How many to show", min_value=10, max_value=5000, value=500, step=50)
    with top[1]:
        only = st.selectbox("Filter", ["ALL", "APPROVE", "BLOCK"], index=0)
    with top[2]:
        q = st.text_input("Search (review_id / analyst / notes)", value="")
    with top[3]:
        if st.button("Refresh", use_container_width=True):
            st.cache_data.clear()

    try:
        data = api_get_feedback_export(limit=int(limit))  # {items:[...], count:N}
        items = data.get("items", []) if isinstance(data, dict) else []
    except Exception as e:
        st.error(f"Failed to fetch closed log: {type(e).__name__}: {e}")
        return

    if not items:
        st.info("No closed tickets yet.")
        return

    df = pd.DataFrame(items)

    if "id" in df.columns and "review_id" not in df.columns:
        df["review_id"] = df["id"]

    if "updated" in df.columns:
        def to_dt(v):
            try:
                if pd.isna(v):
                    return None
                return datetime.fromtimestamp(int(v), tz=timezone.utc).isoformat()
            except Exception:
                return str(v)
        df["updated_dt"] = df["updated"].apply(to_dt)
    else:
        df["updated_dt"] = None

    if only in {"APPROVE", "BLOCK"} and "analyst_decision" in df.columns:
        df = df[df["analyst_decision"] == only]

    if q.strip():
        qq = q.strip().lower()
        def row_match(r):
            for col in ["review_id", "analyst", "notes", "payload_hash"]:
                if col in r and pd.notna(r[col]) and qq in str(r[col]).lower():
                    return True
            return False
        df = df[df.apply(lambda r: row_match(r.to_dict()), axis=1)]

    show_cols = [c for c in [
        "review_id",
        "updated_dt",
        "analyst_decision",
        "analyst",
        "notes",
        "score_xgb",
        "ae_bucket",
        "ae_percentile_vs_legit",
        "reason_codes",
    ] if c in df.columns]

    df_view = df[show_cols].copy()
    if "reason_codes" in df_view.columns:
        df_view["reason_codes"] = df_view["reason_codes"].apply(_fmt_reason_codes)

    st.dataframe(df_view, use_container_width=True, hide_index=True)


def main():
    st.set_page_config(page_title="FPN Review Dashboard", layout="wide")
    cfg = load_config()

    st.title(cfg_get(cfg, "app", "title", default="FPN Review Dashboard"))

    mode, min_score, max_rows = sidebar(cfg)

    page = st.sidebar.radio("Pages", ["Queue", "Log"], index=0)
    if page == "Queue":
        page_queue(cfg, mode=mode, min_score=min_score, max_rows=max_rows)
    else:
        page_log(cfg, mode=mode)


if __name__ == "__main__":
    main()
