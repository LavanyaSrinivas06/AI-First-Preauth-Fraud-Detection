import streamlit as st
import pandas as pd
from dashboard.api import get_review_queue

def render():
    st.title("Review Queue")

    reviews = get_review_queue()
    if not reviews:
        st.success("No transactions pending review.")
        return

    df = pd.DataFrame(reviews)

    st.caption("Only transactions with uncertain risk appear here.")

    selected = st.dataframe(
        df[["review_id", "created_at", "p_xgb", "ae_bucket"]],
        use_container_width=True,
        selection_mode="single-row",
        hide_index=True,
    )

    if selected and selected["selection"]["rows"]:
        idx = selected["selection"]["rows"][0]
        st.session_state["review_id"] = df.iloc[idx]["review_id"]
        st.session_state["page"] = "review"
        st.rerun()
