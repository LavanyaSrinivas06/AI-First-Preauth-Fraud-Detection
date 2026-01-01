# Phase 4 — Feedback Loop (Human-in-the-loop Retraining)

This project implements a thesis-friendly feedback loop for pre-authorization fraud detection:

1) Model produces a decision (APPROVE / REVIEW / BLOCK)
2) REVIEW cases are labeled by a human analyst
3) Labeled reviews are exported into a feedback dataset
4) Offline retraining produces a new XGBoost model artifact
5) A lightweight “promotion” step activates the new model
6) All future decisions remain traceable to the exact model version

---

## 1) Data Flow (textual diagram)

Incoming request (processed 102 features)
→ `/preauth/decision`
→ XGB score + AE score (grey-zone only)
→ decision recorded in SQLite: `decisions`
→ if decision == REVIEW:
   - review row created in SQLite: `reviews`
   - features snapshot saved: `artifacts/snapshots/feature_snapshots/{review_id}.json`
   - minimal payload snapshot saved: `payloads/review/{review_id}.json`

Analyst labels
→ dashboard calls `/reviews/close` (or similar)
→ updates `reviews` row: status=closed + analyst_decision + analyst + notes

Export & retrain (offline)
→ build dataset from closed reviews + feature snapshots:
   `scripts/build_feedback_dataset.py` → `data/processed/feedback_labeled.csv`

→ retrain XGB with base train + feedback:
   `scripts/retrain_from_feedback.py` → `artifacts/models/xgb_model_feedback.pkl`

Promotion (activate model)
→ set `artifacts/models/active_xgb.json` to point to the selected model file:
   {
     "active_model": "xgb_model_feedback.pkl",
     "version": "xgb-feedback-2026w01",
     "created": 1767297095
   }

Inference traceability
→ every decision/review stores `model_version` (e.g., `xgb-feedback-2026w01`)

---

## 2) How feedback becomes labels

- Only CLOSED reviews with an analyst decision are exported.
- Mapping:
  - analyst_decision = BLOCK → label 1 (fraud)
  - analyst_decision = APPROVE → label 0 (legit)

The export uses `feature_path` snapshots to guarantee the model sees the **exact same 102 features** that were scored during inference.

---

## 3) Offline retraining policy (thesis-friendly)

- Retraining is deliberately OFFLINE (weekly / periodic).
- The online API never tunes hyperparameters automatically.
- Retraining script appends feedback data to the base train set (deduped), trains once, and writes:
  - model artifact
  - metrics JSON
  - report markdown

---

## 4) Promotion policy

Promotion is the only step that changes what the API serves.

- A single file (`active_xgb.json`) acts as the registry pointer.
- This preserves reproducibility:
  - old models remain in `artifacts/models/`
  - new models can be activated/reverted by changing the pointer
- All inference rows persist `model_version` so results are explainable later.

---

## 5) Traceability guarantee

For every decision and review:
- `payload_hash` identifies the request payload deterministically
- `feature_path` stores the exact 102 scored features (when REVIEW)
- `model_version` records which model produced the output

This enables audits like:
- “Which model version blocked this transaction?”
- “Compare outcomes before vs after a model promotion”
- “Rollback if a promotion performs worse”
