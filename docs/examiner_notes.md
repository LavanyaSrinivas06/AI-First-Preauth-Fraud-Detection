# Progress update — Phase id (detailed) — January 22, 2026

Dear Professor,

This is a detailed progress update for Phase `id` of the AI-First-Preauth-Fraud-Detection project. I describe every change, exact numeric results from the artifacts, the system design, experiments, and reproduction commands. Where useful I include figures (embedded) so you can see diagnostics directly.

---

## Table of Contents
1. CORE THESIS CONTRIBUTION (HIGH-LEVEL)
2. IMPLEMENTED SYSTEM OVERVIEW
3. MODEL CONTRIBUTIONS
4. HYBRID DECISION LOGIC (KEY CONTRIBUTION)
5. EXPLAINABILITY & ANALYST SUPPORT
6. FEEDBACK LOOP & CONTINUOUS LEARNING
7. CURRENT EVALUATION RESULTS (exact numbers)
8. REPRODUCIBILITY & ARTIFACTS (how to re-run)
9. CURRENT STATUS SUMMARY
10. PLANNED FINAL STEPS

---

## 1. CORE THESIS CONTRIBUTION (HIGH-LEVEL)

Short statement

- The primary contribution is a practical, reproducible hybrid fraud-detection pipeline for pre-authorization that combines:
  - a supervised XGBoost risk scorer tuned for high discriminative performance, and
  - an unsupervised Autoencoder (AE) novelty detector that catches out-of-distribution/novel fraud patterns.

- The hybrid router combines these two signals to: APPROVE safe transactions, BLOCK obvious fraud, and ROUTE gray-zone / anomalous cases to an analyst review queue. Analyst labels are captured and fed back to produce a feedback-augmented XGBoost model.

Why this matters in practice

- Improves safety by reducing over-reliance on a single model class. The supervised model is precise on known patterns; AE protects against novel attacks.
- Everything is saved for auditors: feature snapshots, model versions, SHAP explainability PNGs, and retraining artifacts.

## 2. IMPLEMENTED SYSTEM OVERVIEW

High-level flow (text):

Request → Preprocess → Feature snapshot (102 features JSON) → Model service
  -> XGBoost (p_xgb)
  -> Autoencoder (ae_error)
  -> Hybrid decision router → APPROVE / BLOCK / REVIEW
  -> If REVIEW: Streamlit dashboard for analyst (review row) → analyst decision → feedback event in DB

Key code locations

- API (FastAPI): `api/main.py`, routers under `api/routers/`
- Model service & decision logic: `api/services/model_service.py`
- Store & feedback helpers: `api/services/store.py`
- Explainability helpers: `dashboard/utils/explainability.py`
- Feedback/retrain orchestration: `scripts/feedback_loop.py`
- Demo payload generators: `scripts/generate_*_payloads.py`

 ![Architecture diagram](docs/figures/architecture.svg)

Practical run note

- When running scripts from repo root, use the project's virtualenv and set `PYTHONPATH=.`, for example:

```bash
PYTHONPATH=. .venv/bin/python scripts/feedback_loop.py
```

## 3. MODEL CONTRIBUTIONS

A. XGBoost (supervised)

- Role: main binary classifier for fraud risk (returns `p_xgb`).
- Training hyperparameters used in retrain runs (scripts):

  - n_estimators: 300
  - max_depth: 4
  - learning_rate: 0.05
  - subsample: 0.9
  - colsample_bytree: 0.9
  - reg_lambda: 1.0
  - objective: binary:logistic
  - random_state: 42

- Feature schema: 102 features (preprocessed). Preprocess artifact and feature-order are used to create reproducible snapshots per request.

B. Autoencoder (unsupervised)

- Role: compute reconstruction error (ae_error) used as a novelty signal and to gate reviews/blocks.
- AE outputs include error arrays across baseline legit, validation, and test sets (stored as .npy arrays) and diagnostic histograms.

## 4. HYBRID DECISION LOGIC (KEY CONTRIBUTION)

Design summary

- The router uses both `p_xgb` and `ae_error`.
- Policy used in the demo:
  - If p_xgb < low_threshold → APPROVE
  - If p_xgb >= high_threshold → BLOCK
  - Else (gray zone) → REVIEW; if ae_error >= ae_review_gate escalate review/consider block depending on ae level.

Representative thresholds used in the evaluation and demo runs (exact values):

- XGBoost decision threshold used for pipeline: `0.162`
- Autoencoder review gate (ae_review_gate): `2.8606350898742705`
- Autoencoder block gate (ae_block_gate): `21.662972305300073`

Rationale

- This design reduces blind-spots: the supervised model captures known patterns; the AE captures novelty and provides a second opinion.

## 5. EXPLAINABILITY & ANALYST SUPPORT

What’s available to analysts

- Per-review SHAP local explanations (PNG images) and a debug `.input.json` containing the exact processed feature vector used to generate the explanation.
- Global SHAP summary & bar plots for feature importance.

Robust generation details

- Because XGBoost artifacts and saved metadata sometimes cause TreeExplainer to fail, the SHAP generator implements:
  1. Try TreeExplainer with booster metadata.
 2. If it fails, fallback to `shap.Explainer` with an explicitly constructed masker built from stored feature snapshots.

I save the PNGs and the debug JSON next to them so analysts or auditors can re-run or inspect the exact input used.

Below are example explainability figures (embedded):

![SHAP summary plot](artifacts/explainability/shap/shap_summary.png)

![SHAP local example](artifacts/explainability/shap/shap_local_1.png)

## 6. FEEDBACK LOOP & CONTINUOUS LEARNING

End-to-end implemented flow

1. Analyst closes review with APPROVE/BLOCK in the Streamlit UI. This creates a `feedback_events` row and marks the review closed.
2. Export step reads closed reviews with `feature_path` snapshots and writes `data/processed/feedback_labeled.csv`.
3. Retrain step appends feedback rows to base training set and retrains XGBoost using the hyperparameters above.
4. New model artifact `xgb_model_feedback.pkl` and metrics JSON are written.

Current limitation

- The number of analyst-labeled feedback rows recorded in the artifacts is small (exact count from the last run: `5` feedback rows). This is sufficient to demonstrate the pipeline but not enough to materially change model performance.

## 7. CURRENT EVALUATION RESULTS (exact numbers)

Below are the exact numbers I obtained from the artifacts produced during this work (reported as-is from `artifacts/metrics` and diagnostics produced on January 22, 2026).

### A. XGBoost (representative test set metrics)

- Test ROC AUC: `0.9926583259721296`
- Example operating point (threshold ≈ 0.162):
  - Precision: `0.75`
  - Recall: `0.8076923076923077`
  - Confusion matrix (test):

```
[[42656, 14],
 [10, 42]]
```

### B. Autoencoder (AE) diagnostics

- AE test ROC AUC: `0.9271448144075283`
- AE thresholds used:
  - Review gate: `2.8606350898742705`
  - Block gate: `21.662972305300073`

- AE summary statistics (reconstruction error arrays):
  - Baseline/legit errors, validation, and test arrays were computed and saved; a histogram overlay was generated.

Embedded AE diagnostic histogram:

![AE reconstruction histogram (legit vs test)](artifacts/plots/ae_reconstruction_hist.png)

### C. Feedback-retraining run metrics (example run)

- Feedback rows added in the run: `5`
- Combined rows used for retraining after append: `397965`
- Feedback-augmented model artifact: `artifacts/models/xgb_model_feedback.pkl` and metrics in `artifacts/metrics/xgb_feedback_metrics.json`.

### D. Demo generation counts

- Demo payloads produced during the session:
  - APPROVE: `100` payloads
  - BLOCK: `100` payloads
  - REVIEW: `100` payloads

### E. Feature importance (example)

![XGBoost feature importance](artifacts/plots/xgb_feature_importance.png)

## 8. REPRODUCIBILITY & ARTIFACTS (how to re-run — exact commands)

Prerequisites

- Activate your virtualenv (I used `.venv`). Example:

```bash
source .venv/bin/activate
```

- If scripts error with `ModuleNotFoundError: No module named 'api'`, run with `PYTHONPATH=.` from repo root.

Key reproduction commands I executed (exact):

1) Generate demo payloads (100 each):

```bash
PYTHONPATH=. .venv/bin/python scripts/generate_approve_payloads.py --count 100
PYTHONPATH=. .venv/bin/python scripts/generate_block_payloads.py --count 100
PYTHONPATH=. .venv/bin/python scripts/generate_review_payloads.py --count 100
```

2) Generate SHAP PNGs for reviews (batch):

```bash
PYTHONPATH=. .venv/bin/python scripts/generate_and_store_shap.py
```

3) Run feedback export + retrain (example run that produced the feedback-augmented model):

```bash
PYTHONPATH=. .venv/bin/python scripts/feedback_loop.py
```

4) Produce AE reconstruction histogram and summary (short snippet used in-session):

```bash
.venv/bin/python - <<'PY'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

base = Path('artifacts/ae_errors')
outdir = Path('artifacts/plots')
outdir.mkdir(parents=True, exist_ok=True)

test = np.load(base / 'ae_test_errors.npy')
legit = np.load(base / 'ae_baseline_legit_errors.npy')
plt.figure(figsize=(6,4))
bins = np.histogram(np.hstack([test,legit]), bins=100)[1]
plt.hist(legit, bins=bins, density=True, alpha=0.6, label='legit baseline')
plt.hist(test, bins=bins, density=True, alpha=0.6, label='test set')
plt.axvline(np.percentile(legit,95), color='k', linestyle='--', label='legit 95pct')
plt.xlabel('Reconstruction error')
plt.ylabel('Density')
plt.title('AE reconstruction error: legit vs test')
plt.legend()
plt.tight_layout()
plt.savefig(outdir / 'ae_reconstruction_hist.png', dpi=150)
PY
```

5) Convert report Markdown to DOCX (I added a helper):

```bash
.venv/bin/python scripts/md_to_docx.py docs/phase_id_full_report.md docs/phase_id_full_report.docx
```

## 9. CURRENT STATUS SUMMARY (concise)

- Core system implemented and runnable locally. SHAP explainability generation is robust and persisted per review. Demo payloads (100 each) were created. AE diagnostics and XGBoost metrics were produced. Feedback loop exists end-to-end, but feedback labels are currently sparse (5 rows) so retraining impact is limited.

Immediate blockers: none technical. The main outstanding item is collecting more labeled feedback to demonstrate retraining impact.

## 10. PLANNED FINAL STEPS (short roadmap)

1. Seed or collect ~200+ analyst labels (APPROVE/BLOCK) so feedback-based retraining produces measurable differences.
2. Produce final figures for the thesis: ROC/PR curves with bootstrap 95% CIs, calibration plots, and a combined decision-performance table.
3. Package a reproducibility ZIP with final DOCX, core plots, models, and a README with exact commands; optionally provide a Dockerfile.
4. Add a minimal auditor notebook that re-runs SHAP reproduction for a selected review using the saved `.input.json`.

---

If you prefer, I can now (pick one):

- seed `N` synthetic feedback rows (you pick N and APPROVE/BLOCK split) and run the full feedback-loop to show before/after metrics, or
- bundle a ZIP with the DOCX and the key PNGs for emailing, or
- run bootstrap evaluation and produce ROC/PR plots with confidence intervals for the final thesis figures.

Please tell me which follow-up you want and I will start immediately.

With respect,

[student name]
Examiner notes — likely defense questions and short answers

This file provides concise answers to questions examiners will likely ask during a defense.

1) Reproducibility
- A `REPRODUCE.md` and `scripts/reproduce.py` are provided at the repository root. They outline environment setup and the canonical steps to generate artifacts. Use the provided `requirements.lock.txt` for exact package versions.

2) Model choices
- Supervised: XGBoost chosen for tabular performance and interpretability. Unsupervised: Autoencoder used to detect anomalies (novel fraud patterns). Comparisons with simple baselines (logistic regression) and ablation studies should be provided in the `notebooks/` for completeness.

3) Thresholding and evaluation
- Threshold selection is performed using precision/recall trade-offs and business cost heuristics (see `docs/feedback_retraining_report.md` and artifacts in `artifacts/metrics`). Include ROC/PR plots and cost-matrix justification in the final thesis write-up.

4) Human-in-the-loop
- The system routes uncertain cases to a review queue; analyst decisions are logged and used to create feedback labels for retraining. Ensure reviewer schema and aggregation policy are included in the methodology section.

5) Ethics and privacy
- Document whether data is synthetic or real, anonymization steps, and any limitations. If synthetic data are used, include a short analysis of distributional differences.

6) Limitations to call out
- Any external dependencies for integration tests (databases, third-party APIs) are excluded from unit test runs. Provide clear instructions in REPRODUCE.md on how to run heavy/integration flows.

Suggested quick appendix additions for the thesis document:
- Appendix A: Exact conda/pip environment and `pip freeze` listing.
- Appendix B: Commands and script outputs to reproduce key figures from `docs/`.
