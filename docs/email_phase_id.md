Subject: [Phase: id] Demo / Reproducibility status & key model artifacts

To: [recipient@example.com]
Cc: [team@example.com]

---

Hi team,

Please find a short summary for Phase ID: `id` along with key artifacts (XGBoost + Autoencoder) that you can attach to the demo/repro reproducibility email.

This note highlights model performance, decision thresholds, and where to find the visual artifacts used in the thesis/demo.

## 1) Short summary
Phase ID: id
Project: AI-First-Preauth-Fraud-Detection
Purpose: Demo-ready reproducible pipeline for e2e inference (XGBoost supervised + Autoencoder unsupervised), analyst-triggered SHAP explainability, and beginner-friendly feedback loop for retraining.

## 2) Key model metrics (quick numbers)
- XGBoost (supervised)
  - Test ROC AUC: 0.99266
  - Test precision (example operating point): 0.75
  - Test recall (example operating point): 0.80769
  - Model version used in demo: `xgb-feedback-2026w01`
  - Decision threshold used in pipeline: 0.162 (xgb decision threshold)

- Autoencoder (unsupervised)
  - Test ROC AUC: 0.92714
  - AE thresholds (used for gating / review):
    - review gate: 2.8606350898742705
    - block gate: 21.662972305300073
  - AE reconstruction summary and histogram available under `artifacts/plots/ae_reconstruction_hist.png` and `artifacts/plots/ae_reconstruction_summary.json`.

(These values are pulled from the repo metrics used for the thesis and demo runs.)

## 3) Visual artifacts (please attach these to the email)
- XGBoost feature importance / diagnostic image (example):

  ![XGBoost model](/docs/model/xgboost.png)

- Autoencoder reconstruction / diagnostic image (example):

  ![Autoencoder model](/docs/model/autoencoder.png)

Note: If the exact filenames differ, replace the image paths above with the correct files under `docs/model/`.

## 4) Where to find artifacts in the repo
- Models: `artifacts/models/` (contains `xgb_model.pkl`, `xgb_model_feedback.pkl`, `autoencoder_model.keras`)
- Model metrics: `artifacts/metrics/xgb_metrics.json`, `artifacts/metrics/ae_metrics.json`, `artifacts/metrics/xgb_feedback_metrics.json`
- AE errors and histogram: `artifacts/ae_errors/` and `artifacts/plots/ae_reconstruction_hist.png`
- Explainability (SHAP) images: `artifacts/explainability/shap/` (also `dashboard/static/shap/` may contain PNGs used by the Streamlit UI)
- Demo payloads: `demo_payloads/{approve,block_fraud,review_legit}/`

## 5) Suggested email body (copy / paste)

Subject: Demo package — Phase `id` (AI-First-Preauth-Fraud-Detection)

Hi all,

Attached are the key artifacts for Phase `id` of the preauth fraud-detection demo. The package includes:
- XGBoost model diagnostics and metrics (AUC ~0.993)
- Autoencoder diagnostics and AE reconstruction histogram (AUC ~0.927)
- A set of demo payloads (100 per decision class) to run locally

Please find the primary images attached (`xgboost.png`, `autoencoder.png`) and the short report with exact metrics in `artifacts/metrics/`.

If you'd like, I can also attach the `data/processed/feedback_labeled.csv` and run `scripts/feedback_loop.py` to produce the feedback-augmented model for a live demo.

Thanks,
[Your name]

## 6) Quick instructions to convert this note to PDF (optional)
If you prefer to attach a PDF rather than a markdown file, use pandoc (if installed):

```bash
# from repo root
pandoc docs/email_phase_id.md -o docs/email_phase_id.pdf --pdf-engine=xelatex
```

Alternatively, open the Markdown in VS Code and export to PDF or copy into your mail client and attach the images listed above.

---

If you want, I can:
- verify the exact image filenames under `docs/model` and update this document to point to the correct names,
- produce a PDF and attach the two images into a single ZIP for easy emailing,
- or create a small `scripts/seed_feedback_demo.py` to add demo feedback and then run `scripts/feedback_loop.py` so you can show retraining live.

Which of those follow-ups do you want me to do next?  
