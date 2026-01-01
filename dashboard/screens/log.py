from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

import pandas as pd
import streamlit as st

from dashboard.utils.api_client import api_get_feedback_export
from dashboard.utils.formatters import fmt_reason_codes


def render_log_page(cfg: Dict[str, Any], *, api_base: str):
    st.subheader("Closed Tickets Log (APPROVE / BLOCK)")

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
            st.rerun()

    try:
        data = api_get_feedback_export(api_base, limit=int(limit))
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
        def to_dt(v: Any):
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
            for col in ["review_id", "analyst", "notes", "payload_hash", "model_version"]:
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
        "model_version",
        "reason_codes",
    ] if c in df.columns]

    df_view = df[show_cols].copy()
    if "reason_codes" in df_view.columns:
        df_view["reason_codes"] = df_view["reason_codes"].apply(fmt_reason_codes)

    st.dataframe(df_view, use_container_width=True, hide_index=True)
