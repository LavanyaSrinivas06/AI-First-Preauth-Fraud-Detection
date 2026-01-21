import os
import streamlit as st
from pages import home, review_queue, review_detail

st.set_page_config(
    page_title="AI-First Pre-Authorization Fraud Review",
    layout="wide",
)

# Allow overriding the API base via environment variable FPN_API_BASE
api_base = os.environ.get("FPN_API_BASE", "http://127.0.0.1:8000")

page = st.session_state.get("page", "home")

if page == "home":
    home.render()

elif page == "queue":
    review_queue.render(api_base=api_base)

elif page == "review_detail":
    review_detail.render(api_base=api_base)

else:
    st.error(f"Unknown page: {page}")
