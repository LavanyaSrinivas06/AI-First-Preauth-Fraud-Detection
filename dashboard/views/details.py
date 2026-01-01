from __future__ import annotations

from typing import Any, Dict, Callable, Optional

import streamlit as st

from dashboard.utils.formatters import fmt_reason_details
from dashboard.utils.shap_utils import (
    shap_png_path_for_review,
    generate_shap_png,
    terminal_command_for_shap,
)


def render_details(
    cfg: Dict[str, Any],
    *,
    review: Dict[str, Any],
    static_dir: str,
    on_back: Optional[Callable[[], None]] = None,
):
    review_id = str(review.get("id", ""))
    st.markdown("---")
    st.subheader(f"Details: {review_id}")

    # Why
    st.markdown("### Why it was sent to REVIEW")
    details = review.get("reason_details")
    if details:
        for line in fmt_reason_details(details):
            st.write(f"- {line}")
    else:
        rc = review.get("reason_codes", [])
        if isinstance(rc, list) and rc:
            for r in rc:
                st.write(f"- {r}")
        else:
            st.write("- Gray-zone / anomaly gate.")

    # Scores
    st.markdown("### Scores")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("XGB probability", f"{review.get('score_xgb', '—')}")
    c2.metric("AE bucket", f"{review.get('ae_bucket', '—')}")
    pct = review.get("ae_percentile_vs_legit", None)
    c3.metric("AE percentile vs legit", (f"{float(pct):.2f}" if pct is not None else "—"))
    c4.metric("Model version", f"{review.get('model_version', '—')}")

    # Payload snapshot
    with st.expander("Payload snapshot (stored subset)", expanded=False):
        pm = review.get("payload_min", None)
        if isinstance(pm, dict) and pm:
            st.json(pm)
        else:
            st.info("No payload_min found for this review.")

    # SHAP
    st.markdown("### SHAP explanation")

    shap_path = shap_png_path_for_review(static_dir, review_id)
    shap_file = None
    if shap_path and __import__("pathlib").Path(shap_path).exists():
        shap_file = shap_path

    left, right = st.columns([1.2, 1.8], vertical_alignment="top")
    with left:
        if st.button("Generate SHAP now", use_container_width=True):
            ok, msg = generate_shap_png(
                review=review,
                static_dir=static_dir,
                artifacts_dir="artifacts",
                sqlite_path="artifacts/stores/inference_store.sqlite",
                xgb_model_relpath="models/xgb_model.pkl",
                max_display=15,
            )
            if ok:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)

        st.caption("Or generate from terminal:")
        st.code(terminal_command_for_shap(review_id, out_dir=static_dir), language="bash")

    with right:
        if shap_file and __import__("pathlib").Path(shap_file).exists():
            st.image(shap_file, caption=f"SHAP for {review_id}", use_container_width=True)
        else:
            st.warning("Per-review SHAP not found yet.")

    if on_back is not None:
        if st.button("Back to queue"):
            on_back()
            st.rerun()
