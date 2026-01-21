import streamlit as st
import pandas as pd
from utils.api_client import api_get_review_queue

def render(api_base: str):
    st.title("Review Queue")

    view = st.radio(
        "Category",
        ["Open Reviews", "Approved", "Blocked"],
        horizontal=True,
    )

    try:
        items = api_get_review_queue(api_base)
    except Exception as e:
        st.error(f"Failed to load review queue: {e}")
        return

    if not items:
        st.info("No reviews available.")
        return

    df = pd.DataFrame(items)

    # OPEN = analyst_decision is null
    if view == "Open Reviews":
        df = df[df["analyst_decision"].isna()]
    elif view == "Approved":
        df = df[df["analyst_decision"] == "APPROVE"]
    elif view == "Blocked":
        df = df[df["analyst_decision"] == "BLOCK"]

    st.caption(f"{len(df)} transaction(s)")

    for _, r in df.iterrows():
        with st.container(border=True):
            c1, c2, c3, c4, c5 = st.columns([2.5, 1.5, 1.5, 3.5, 1.5])

            c1.write(r["id"])
            c2.write(f"{r['score_xgb']:.3f}")
            c3.write(r["ae_bucket"])
            c4.write(", ".join(r.get("reason_codes", [])) or "—")

            if view == "Open Reviews":
                if c5.button("View", key=r["id"]):
                    st.session_state["selected_review_id"] = r["id"]
                    st.session_state["page"] = "review_detail"
                    st.rerun()

    if st.button("← Back to Home"):
        st.session_state["page"] = "home"
        st.rerun()
