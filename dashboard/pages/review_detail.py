import streamlit as st
from utils.api_client import api_get_review, api_close_review
from utils.explainability import generate_shap, shap_path

def render(api_base: str):
    review_id = st.session_state.get("selected_review_id")
    if not review_id:
        st.error("No review selected.")
        return

    review = api_get_review(api_base, review_id)

    st.title("Transaction Review")
    st.subheader(f"Review ID: {review_id}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Fraud Probability (XGBoost)", f"{review['score_xgb']:.3f}")
    c2.metric("AE Risk Bucket", review["ae_bucket"])
    c3.metric("AE Percentile vs Legit", f"{review['ae_percentile_vs_legit']:.2f}%")

    st.divider()

    st.subheader("Why this transaction requires review")
    st.write(
        "The transaction falls into an uncertainty region where "
        "the AI models cannot confidently approve or block it."
    )

    st.divider()

    st.subheader("Explainability (SHAP)")

    if st.button("Generate SHAP Explanation"):
        generate_shap(review)
        st.success("SHAP explanation generated.")

    shp = shap_path(review_id)
    if shp.exists():
        col_img, col_txt = st.columns([3, 2])
        with col_img:
            st.image(str(shp), use_container_width=True)

        # try to show top-5 SHAP features from the debug input.json written by generate_shap_png
        try:
            dbg = shp.with_suffix('.input.json')
            if dbg.exists():
                import json
                data = json.loads(dbg.read_text(encoding='utf-8'))
                top_pos = data.get('top_positive', []) or []
                top_neg = data.get('top_negative', []) or []
                with col_txt:
                    st.markdown('**Top positive contributors**')
                    if top_pos:
                        for t in top_pos:
                            st.write(f"{t['feature']}: {t['value']:.4f}")
                    else:
                        st.write('—')

                    st.markdown('**Top negative contributors**')
                    if top_neg:
                        for t in top_neg:
                            st.write(f"{t['feature']}: {t['value']:.4f}")
                    else:
                        st.write('—')
        except Exception:
            # don't crash UI if debug file parse fails
            pass

    st.divider()

    st.subheader("Final decision")

    analyst = st.text_input("Analyst ID")
    decision = st.radio("Decision", ["APPROVE", "BLOCK"])
    notes = st.text_area("Decision notes (optional)")

    if st.button("Submit decision"):
        api_close_review(api_base, review_id, analyst, decision, notes)
        st.success(f"Review closed as {decision}")
        st.session_state["page"] = "queue"
        st.rerun()

    if st.button("← Back to Review Queue"):
        st.session_state["page"] = "queue"
        st.rerun()
