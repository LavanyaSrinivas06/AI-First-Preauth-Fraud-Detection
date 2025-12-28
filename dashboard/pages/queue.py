from __future__ import annotations

from typing import Any, Dict, Optional
import pandas as pd
import streamlit as st

from dashboard.utils.api_client import api_get_review_queue, api_get_review
from dashboard.utils.formatters import fmt_dt, fmt_reason_codes
from dashboard.components.details import render_details


def _cfg_get(cfg: Dict[str, Any], *keys: str, default=None):
    cur: Any = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _set_selected(review_id: str) -> None:
    st.session_state["selected_review_id"] = review_id


def _clear_selected() -> None:
    st.session_state.pop("selected_review_id", None)


def _get_selected() -> Optional[str]:
    return st.session_state.get("selected_review_id")


def render_queue_page(cfg: Dict[str, Any], *, api_base: str, min_score: float, max_rows: int):
    st.subheader("Review Queue (Open)")

    static_dir = str(_cfg_get(cfg, "paths", "static_dir", default="dashboard/static"))

    items = api_get_review_queue(api_base) or []
    if max_rows:
        items = items[: int(max_rows)]

    df = pd.DataFrame(items)
    if df.empty:
        st.info("No open REVIEW tickets. Trigger /preauth/decision that returns REVIEW.")
        return

    if "score_xgb" in df.columns:
        df["score_xgb"] = pd.to_numeric(df["score_xgb"], errors="coerce")
        df = df[df["score_xgb"].fillna(0.0) >= float(min_score)]

    if df.empty:
        st.info("No tickets match the current filter.")
        return

    st.caption("Click View to open details inline.")

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
        ts = fmt_dt(row.get("timestamp", None))
        why = fmt_reason_codes(row.get("reason_codes", None))

        c1, c2, c3, c4 = st.columns([2.2, 2.6, 4.8, 1.2])
        with c1:
            st.write(review_id)
        with c2:
            st.write(ts)
        with c3:
            st.write(why)
        with c4:
            if st.button("View", key=f"view_{review_id}_{i}", use_container_width=True):
                _set_selected(review_id)

    sel = _get_selected()
    if sel:
        try:
            review = api_get_review(api_base, sel)
        except Exception as e:
            st.error(f"Failed to fetch review: {type(e).__name__}: {e}")
            return

        render_details(
            cfg,
            review=review,
            static_dir=static_dir,
            on_back=_clear_selected,
        )
