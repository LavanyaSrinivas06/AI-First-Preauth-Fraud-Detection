import streamlit as st
import pandas as pd


def render():
    st.title("Model Performance Overview")

    # --------------------------------------------------
    # TEMPORARY DATA SOURCE (for wiring & UI validation)
    # --------------------------------------------------
    # This will be replaced with SQLite loading next
    df = pd.DataFrame({
        "p_xgb": [0.02, 0.12, 0.38, 0.55, 0.82, 0.91],
        "decision": ["APPROVE", "APPROVE", "REVIEW", "REVIEW", "BLOCK", "BLOCK"],
        "ae_bucket": ["normal", "normal", "elevated", "normal", "extreme", "extreme"],
        "review_final_decision": ["APPROVE", None, "APPROVE", "BLOCK", None, None],
    })

    t_low = 0.05
    t_high = 0.80

    # ------------------ XGBOOST ------------------
    st.subheader("Supervised Model Health (XGBoost)")

    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Fraud Probability", f"{df['p_xgb'].mean():.3f}")
    c2.metric(
        "Gray-Zone Rate",
        f"{((df['p_xgb'] >= t_low) & (df['p_xgb'] < t_high)).mean() * 100:.1f}%"
    )
    c3.metric(
        "High-Risk Rate",
        f"{(df['p_xgb'] >= t_high).mean() * 100:.1f}%"
    )

    st.bar_chart(df["p_xgb"])

    st.divider()

    # ------------------ AUTOENCODER ------------------
    st.subheader("Unsupervised Model Health (Autoencoder)")

    ae_dist = df["ae_bucket"].value_counts(normalize=True)
    st.bar_chart(ae_dist)

    st.divider()

    # ------------------ DECISION FLOW ------------------
    st.subheader("Decision Flow Efficiency")

    decision_dist = df["decision"].value_counts(normalize=True)
    st.bar_chart(decision_dist)

    auto_rate = (df["decision"] != "REVIEW").mean()
    st.metric("Automated Decision Rate", f"{auto_rate * 100:.1f}%")

    st.divider()

    # ------------------ HUMAN REVIEW ------------------
    st.subheader("Human Review Outcomes")

    reviews = df[df["decision"] == "REVIEW"]
    if len(reviews) > 0:
        c1, c2 = st.columns(2)
        c1.metric(
            "Human Approval Rate",
            f"{(reviews['review_final_decision'] == 'APPROVE').mean() * 100:.1f}%"
        )
        c2.metric(
            "Human Block Rate",
            f"{(reviews['review_final_decision'] == 'BLOCK').mean() * 100:.1f}%"
        )
    else:
        st.info("No reviews completed yet.")
