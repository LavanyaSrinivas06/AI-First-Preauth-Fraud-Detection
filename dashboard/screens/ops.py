from __future__ import annotations

from typing import Any, Dict

import streamlit as st

from dashboard.utils.api_client import (
    api_get_feedback_export,
    api_get_feedback_summary,
    api_get_health_model,
    api_get_review_queue,
)


def render_ops_page(cfg: Dict[str, Any], *, api_base: str):
    st.subheader("Ops Overview")

    left, right = st.columns([1.4, 1.0], vertical_alignment="top")

    # --- KPIs ---
    with left:
        c1, c2, c3, c4 = st.columns(4)

        # Active model card
        active_model = "—"
        model_version = "—"
        promoted_at = "—"
        try:
            hm = api_get_health_model(api_base)
            model_version = hm.get("model_version") or hm.get("version") or "—"
            active_model = hm.get("active_model") or hm.get("xgb_model") or "—"
            promoted_at = hm.get("registry_created") or hm.get("created") or "—"
        except Exception:
            pass

        # Queue size
        open_queue = "—"
        try:
            q = api_get_review_queue(api_base)
            open_queue = str(len(q))
        except Exception:
            pass

        # Closed count (from feedback export)
        closed_count = "—"
        try:
            log = api_get_feedback_export(api_base, limit=5000)
            items = log.get("items", []) if isinstance(log, dict) else []
            closed_count = str(len(items))
        except Exception:
            pass

        # Feedback summary (optional endpoint)
        fb_total = "—"
        try:
            s = api_get_feedback_summary(api_base)
            fb_total = str(s.get("feedback_total", "—"))
        except Exception:
            pass

        c1.metric("Active model version", model_version)
        c2.metric("Active model file", active_model)
        c3.metric("Open reviews", open_queue)
        c4.metric("Closed reviews", closed_count)

        st.caption("If some cards show '—', it means that endpoint isn't available yet (still OK).")

        st.markdown("---")

        st.markdown("### Quick Actions")
        st.write("- Go to **Queue** to assign and close open reviews.")
        st.write("- Go to **Log** to review closed tickets and labels collected for retraining.")

    # --- Right panel: status + notes ---
    with right:
        st.markdown("### System Status")
        try:
            hm = api_get_health_model(api_base)
            st.success("API reachable")
            st.json(hm)
        except Exception as e:
            st.error(f"API not reachable: {type(e).__name__}: {e}")

        st.markdown("### Notes")
        st.info(
            "Model version should stay constant during a run period (e.g., the whole week). "
            "Only change it when you promote a validated model via `artifacts/models/active_xgb.json`."
        )
