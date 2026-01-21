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
