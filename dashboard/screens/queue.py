from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import streamlit as st

from dashboard.utils.api_client import api_assign_review, api_close_review, api_get_review, api_get_review_queue
from dashboard.utils.formatters import fmt_reason_codes, fmt_reason_details
from dashboard.utils.shap_utils import generate_shap_png, shap_png_path_for_review, terminal_command_for_shap


def _ts_to_iso(v: Any) -> str:
    if v is None or v == "":
        return "—"
    try:
        if isinstance(v, int):
            return datetime.fromtimestamp(v, tz=timezone.utc).isoformat()
        return str(v)
    except Exception:
        return str(v)


def render_queue_page(cfg: Dict[str, Any], *, api_base: str):
    st.subheader("Open Review Queue")

    # Top action bar
    bar = st.columns([1.2, 1.4, 1.0, 1.0])
    with bar[0]:
        analyst = st.text_input("Analyst (your id/email)", value=st.session_state.get("analyst", "analyst@fpn"))
        st.session_state["analyst"] = analyst
    with bar[1]:
        query = st.text_input("Search (review_id / payload_hash)", value="")
    with bar[2]:
        only_unassigned = st.checkbox("Only unassigned", value=False)
    with bar[3]:
        if st.button("Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # Load queue
    try:
        items = api_get_review_queue(api_base)
    except Exception as e:
        st.error(f"Failed to fetch queue: {type(e).__name__}: {e}")
        return

    if not items:
        st.info("No open reviews in the queue.")
        return

    # Filter
    filtered: List[Dict[str, Any]] = []
    q = query.strip().lower()
    for r in items:
        rid = str(r.get("id", "")).lower()
        ph = str(r.get("payload_hash", "")).lower()
        assigned = str(r.get("analyst") or "").strip()
        if q and (q not in rid) and (q not in ph):
            continue
        if only_unassigned and assigned:
            continue
        filtered.append(r)

    st.caption(f"Showing {len(filtered)} of {len(items)} open reviews")

    # Table-like rows
    header = st.columns([2.2, 2.0, 3.0, 1.5, 1.1])
    header[0].markdown("**Review ID**")
    header[1].markdown("**Time**")
    header[2].markdown("**Why review**")
    header[3].markdown("**Assigned**")
    header[4].markdown("**Action**")
    st.divider()

    for r in filtered[:200]:
        rid = str(r.get("id", ""))
        created = _ts_to_iso(r.get("created"))
        why = fmt_reason_codes(r.get("reason_codes"))
        assigned = (r.get("analyst") or "").strip() or "—"

        row = st.columns([2.2, 2.0, 3.0, 1.5, 1.1])
        row[0].write(rid)
        row[1].write(created)
        row[2].write(why)
        row[3].write(assigned)

        if row[4].button("View", key=f"view_{rid}"):
            st.session_state["selected_review_id"] = rid
            st.rerun()

    # Details drawer
    sel = st.session_state.get("selected_review_id")
    if not sel:
        return

    st.markdown("---")
    st.subheader(f"Details: {sel}")

    try:
        review = api_get_review(api_base, sel)
    except Exception as e:
        st.error(f"Failed to fetch review: {type(e).__name__}: {e}")
        return

    # Summary cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("XGB probability", f"{review.get('score_xgb', '—')}")
    c2.metric("AE bucket", f"{review.get('ae_bucket', '—')}")
    pct = review.get("ae_percentile_vs_legit", None)
    c3.metric("AE percentile", (f"{float(pct):.2f}" if pct is not None else "—"))
    c4.metric("Model version", f"{review.get('model_version', '—')}")

    # Why review
    st.markdown("### Why it was sent to REVIEW")
    details = review.get("reason_details")
    if details:
        for line in fmt_reason_details(details):
            st.write(f"- {line}")
    else:
        rc = review.get("reason_codes", [])
        if isinstance(rc, list) and rc:
            for x in rc:
                st.write(f"- {x}")
        else:
            st.write("- Gray-zone / anomaly gate")

    # Payload snapshot
    with st.expander("Payload snapshot (stored subset)", expanded=False):
        pm = review.get("payload_min", None)
        if isinstance(pm, dict) and pm:
            st.json(pm)
        else:
            st.info("No payload_min found.")

    # Assignment + close actions
    st.markdown("### Actions")

    a1, a2, a3 = st.columns([1.2, 1.2, 2.2], vertical_alignment="top")

    with a1:
        if st.button("Assign to me", use_container_width=True):
            try:
                api_assign_review(api_base, sel, analyst=analyst)
                st.success("Assigned.")
                st.cache_data.clear()
                st.rerun()
            except Exception as e:
                st.error(
                    "Assign failed. If you haven't added the API endpoint yet, do that next.\n"
                    f"{type(e).__name__}: {e}"
                )

    with a2:
        close_decision = st.selectbox("Close as", ["APPROVE", "BLOCK"], index=0)
        notes = st.text_area("Notes", value="", height=90)
        if st.button("Close review", use_container_width=True):
            try:
                api_close_review(api_base, sel, analyst=analyst, decision=close_decision, notes=notes)
                st.success(f"Closed as {close_decision}.")
                st.session_state["selected_review_id"] = None
                st.cache_data.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Close failed: {type(e).__name__}: {e}")

    # SHAP
    with a3:
        st.markdown("**SHAP explanation**")
        static_dir = "dashboard/static"
        shap_path = shap_png_path_for_review(static_dir, sel)

        b1, b2 = st.columns([1.0, 1.2])
        with b1:
            if st.button("Generate SHAP", use_container_width=True):
                ok, msg = generate_shap_png(review=review, static_dir=static_dir)
                if ok:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
        with b2:
            st.caption("Terminal option:")
            st.code(terminal_command_for_shap(sel, out_dir=static_dir), language="bash")

        p = __import__("pathlib").Path(shap_path)
        if p.exists():
            st.image(str(p), caption=f"SHAP for {sel}", use_container_width=True)
        else:
            st.info("No SHAP image yet for this review.")

    if st.button("Back to queue"):
        st.session_state["selected_review_id"] = None
        st.rerun()
