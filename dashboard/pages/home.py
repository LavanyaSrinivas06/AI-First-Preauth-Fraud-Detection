import streamlit as st

def render():
    st.title("AI-First Pre-Authorization Fraud Review Dashboard")

    st.markdown(
        """
        This dashboard demonstrates an **AI-first fraud detection system**
        operating **before payment authorization**.

        Transactions are:
        - ✅ **Approved** automatically when risk is low
        - ❌ **Blocked** automatically when risk is high
        - 🔍 **Reviewed** by analysts when risk is uncertain
        """
    )

    st.divider()

    c1, c2, c3 = st.columns(3)
    c1.metric("Approve", "Low Risk")
    c2.metric("Review", "Uncertain")
    c3.metric("Block", "High Risk")

    st.divider()

    if st.button("🔍 Open Review Queue", use_container_width=True):
        st.session_state["page"] = "queue"
        st.rerun()
