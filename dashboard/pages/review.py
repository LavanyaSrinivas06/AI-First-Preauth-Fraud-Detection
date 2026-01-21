import streamlit as st
from dashboard.api import get_review, submit_feedback

def render():
    review_id = st.session_state.get("review_id")
    if not review_id:
        st.warning("No review selected.")
        return

    review = get_review(review_id)

    st.title(f"Transaction Review: {review_id}")

    # --- Scores ---
    c1, c2, c3 = st.columns(3)
    c1.metric("Fraud Probability (XGB)", f"{review['p_xgb']:.3f}")
    c2.metric("AE Bucket", review["ae_bucket"])
    c3.metric("AE Percentile", f"{review['ae_percentile']:.1f}")

    st.divider()

    # --- Explanation ---
    st.subheader("Why this transaction requires review")

    for r in review["reason_details"]:
        st.markdown(f"**{r['code'].replace('_',' ').title()}**")
        st.write(r["message"])

        if "signals" in r:
            for s in r["signals"]:
                st.markdown(f"- {s}")

    st.divider()

    # --- Analyst decision ---
    st.subheader("Analyst Decision")

    comment = st.text_area("Notes (optional)")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("✅ Approve Transaction"):
            submit_feedback(review_id, "APPROVE", comment)
            st.success("Transaction approved.")
            st.session_state["page"] = "queue"
            st.rerun()

    with c2:
        if st.button("❌ Confirm Fraud"):
            submit_feedback(review_id, "BLOCK", comment)
            st.success("Fraud confirmed.")
            st.session_state["page"] = "queue"
            st.rerun()
