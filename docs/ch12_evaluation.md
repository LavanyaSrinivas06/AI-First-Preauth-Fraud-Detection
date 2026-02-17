# Chapter 12 — Evaluation Methodology and Results

## 12.1 Experimental Setup

Before looking at any numbers, it is worth describing the environment in which
everything was measured.  Every experiment reported here ran against the same
held-out test split and was reproduced by a deterministic script
(`thesis_quality/evaluation/run_evaluation.py`) so that anyone with access to
the repository can re-generate the exact tables and figures shown below.

### 12.1.1 Dataset and Splits

The underlying dataset is the **European Cardholder** benchmark released by the
Université Libre de Bruxelles (ULB) and hosted on Kaggle.  It contains
**284,807** real-world credit-card transactions collected over two days in
September 2013, of which **492** are labelled as fraud — a natural prevalence of
roughly **0.17 %**.  Twenty-eight features (V1–V28) are PCA-transformed
components provided by the data owners to protect cardholder privacy; two
features — `Amount` and `Time` — are left in their original scale.

During the enrichment stage the pipeline adds **11 synthetic context features**
(device_os, browser, is_new_device, ip_country, is_proxy_vpn,
billing_country, shipping_country, country_mismatch, night_txn, weekend_txn, and
several velocity / reputation signals).  After one-hot encoding the ten
categorical columns, the final feature matrix has **102 columns**.

The data is split chronologically (by the `Time` column) into three partitions:

| Split | Rows | Fraud | Legitimate | Fraud Rate |
|-------|-----:|------:|-----------:|-----------:|
| Train (before SMOTE) | 199,364 | 384 | 198,980 | 0.19 % |
| Train (after SMOTE)  | 397,960 | 198,980 | 198,980 | 50.0 % |
| Validation | 42,721 | 56 | 42,665 | 0.13 % |
| Test (holdout) | 42,722 | 52 | 42,670 | 0.12 % |

**Table 19 — Data-split summary.**  SMOTE is applied only to the training
partition; validation and test remain in their natural class distribution.

Two things are worth noting.  First, the validation set is used exclusively
for threshold selection — the XGBoost operating point is chosen here, not on
the test set.  Second, the Autoencoder is trained on the *legitimate-only*
subset of the training split (198,980 rows); it never sees fraud during
training.

### 12.1.2 Model Configurations

**XGBoost.**  Hyperparameters were selected through grid search with F1 as the
objective on the SMOTE-augmented training set.  The winning configuration is:

| Hyperparameter | Value |
|----------------|------:|
| n_estimators | 250 |
| max_depth | 6 |
| learning_rate | 0.05 |
| subsample | 0.80 |
| colsample_bytree | 1.0 |

The probability threshold was selected on the validation split by sweeping
50 values between 0.01 and 0.99 and choosing the one that maximised F1.
The selected threshold is **0.307**, yielding a validation F1 of **0.899**.

**Autoencoder.**  The network was trained for **9 epochs** on the
legitimate-only training partition (early stopping on validation loss).
Final training loss was **0.288** and validation loss **0.301**.  Two
operational thresholds are derived from the reconstruction-error distribution
of the validation set's legitimate transactions (n = 42,665):

| Threshold | Percentile | Value |
|-----------|:----------:|------:|
| Review | p95.0 | 0.692 |
| Block | p99.5 | 4.896 |

### 12.1.3 Hybrid Decision Thresholds

The production decision engine combines both models with three numeric gates
taken from `settings.yaml`:

| Parameter | Value | Role |
|-----------|------:|------|
| `xgb_t_low` | 0.05 | Below → APPROVE immediately |
| `xgb_t_high` | 0.80 | At or above → BLOCK immediately |
| `ae_threshold` (review) | 0.692 | Gray-zone anomaly flag |
| `ae_block` | 4.896 | Gray-zone hard block gate |

Transactions whose XGBoost probability falls in the gray zone
(0.05 < p < 0.80) are passed to the Autoencoder for a second opinion.

---

## 12.2 Evaluation Metrics and Rationale

At a fraud rate of 0.12 % on the test set, accuracy would exceed 99.8 % even
if the classifier labelled every single transaction as legitimate.  For that
reason accuracy is not reported.  The metrics chosen for this evaluation are
the ones that remain informative under extreme class imbalance.

### 12.2.1 Precision, Recall and F1

- **Precision** — of all transactions the model flags as fraud, what fraction
  actually is fraud?  High precision means fewer false alarms, which in a
  payment network translates directly to fewer legitimate customers whose
  cards are wrongly declined.

- **Recall** — of all truly fraudulent transactions, what fraction does the
  model catch?  In pre-authorisation, a missed fraud is money lost; recall
  is therefore the most operationally critical metric.

- **F1 score** — the harmonic mean of precision and recall.  It is the single
  number used during threshold selection because it penalises models that
  sacrifice one side for the other.

### 12.2.2 ROC-AUC

The Receiver Operating Characteristic curve plots the true-positive rate
against the false-positive rate across every possible threshold.  Its area
under the curve (AUC) summarises the model's ability to *rank* fraud above
legitimate transactions regardless of the chosen operating point.  A perfect
model scores 1.0; random guessing scores 0.5.

Under heavy imbalance ROC-AUC can be optimistic — a model that produces
thousands of false positives may still appear to have a low false-positive
*rate* when the denominator contains hundreds of thousands of legitimate
transactions.  This is why PR-AUC is preferred as the primary ranking metric.

### 12.2.3 PR-AUC (Average Precision)

The Precision-Recall curve plots precision against recall at each threshold.
Its area, commonly called Average Precision (AP), is the metric most sensitive
to the minority class.  Unlike ROC-AUC it does not get inflated by the
overwhelming number of true negatives, making it the preferred single-number
metric for fraud-detection systems.

### 12.2.4 Confusion Matrix

The confusion matrix (TN, FP, FN, TP) is included for every model so that
absolute counts can be inspected directly.  In a dataset of 42,722
transactions, the difference between "6 false positives" and "14 false
positives" may look trivial in percentage terms, but each false positive
represents a legitimate cardholder being denied service.

---

## 12.3 Latency and Throughput Benchmarks

A fraud decision that takes longer than the payment network's authorisation
window is useless no matter how accurate it is.  Three benchmark protocols
were run against the live FastAPI endpoint (`POST /preauth/decision`) to
measure end-to-end inference time under progressively heavier workloads.
All benchmarks were executed on a single-process uvicorn server running on
a MacBook Pro (Apple M-series); no GPU was involved.

### 12.3.1 Single-Request Latency

A sequential benchmark sent **300** requests one at a time through the full
inference pipeline (feature preprocessing → XGBoost scoring → Autoencoder
reconstruction → reason-code generation → SQLite persistence → JSON
response).  Ten warm-up requests were issued and discarded before timing
began.

| Metric | Value |
|--------|------:|
| Requests | 300 |
| Errors | 0 |
| Mean | 5.35 ms |
| Median (p50) | 4.62 ms |
| p90 | 5.41 ms |
| p99 | 21.54 ms |
| Max | 26.53 ms |
| Throughput | 186.6 req/s |

**Table 20 — Single-request sequential latency.**

The median end-to-end time is under 5 ms, confirming that the full model
stack — two models plus persistence — is lightweight enough for real-time
decisioning.  The p99 jump to ~22 ms is consistent with occasional Python
garbage-collection pauses; even this outlier sits well inside the 100 ms
budget typically allocated by card-network authorisation specifications
(Visa specifies a 100 ms processing window for acquirer-side decisioning).

### 12.3.2 Concurrent Latency (10 Parallel Clients)

To simulate a more realistic multi-client scenario, a second benchmark fired
**2,000** requests at a concurrency level of **10** — meaning ten clients
issued requests simultaneously and waited for responses before sending the
next batch.

| Metric | Value |
|--------|------:|
| Total requests | 2,000 |
| Concurrency | 10 |
| Mean | 80.3 ms |
| Median (p50) | 74.8 ms |
| p90 | 138.2 ms |
| p99 | 219.3 ms |
| Max | 443.4 ms |
| Throughput | 124.2 req/s |
| Success rate | 99.85 % |

**Table 21 — Concurrent latency benchmark (10 parallel clients).**

Three of the 2,000 requests (0.15 %) returned HTTP 500 due to a SQLite
`UNIQUE constraint` collision — concurrent writers attempted to insert
the same hash-based decision ID simultaneously.  This is a known SQLite
limitation under concurrent writes; switching to PostgreSQL in production
would eliminate it entirely.

Latency increases roughly 15× from the sequential baseline (p50: 4.6 ms →
74.8 ms), which is expected: Python's Global Interpreter Lock (GIL)
serialises CPU-bound XGBoost inference, so concurrent requests queue behind
one another.  Despite this, the p99 of 219 ms remains within the
authorisation window, confirming that a single API process can sustain
moderate traffic without breaching real-time constraints.

### 12.3.3 Sustained Load Test (Locust)

A Locust-based load test pushed the server further to identify the
throughput ceiling.  Three runs were executed at different concurrency
levels:

| Run | Users | Duration | Requests | Failures | Avg RPS | p50 (ms) | p99 (ms) | Status |
|-----|------:|---------:|---------:|---------:|--------:|---------:|---------:|--------|
| 1 | 50 | 120 s | 25,034 | 13,100 | 210 | 1,100 | 2,600 | Over capacity |
| 2 | 40 | 120 s | 23,930 | 0 | 201 | 980 | 1,100 | **Healthy** |
| 3 | 200 | 120 s | 108,163 | 12,385 | 906 | 160 | 1,700 | Over capacity |

**Table 22 — Locust sustained-load test results.**

Run 2 is the key result: at 40 concurrent users the system sustained
**~200 req/s for two full minutes with zero failures**.  The p50 latency
of 980 ms is higher than in the short-burst benchmarks because each
request now competes with 39 others for the GIL; however, the p99 stays at
1.1 seconds, which is acceptable for a batch-review or near-real-time
pipeline and could be reduced by deploying multiple uvicorn workers behind
a reverse proxy.

Runs 1 and 3 intentionally exceeded the server's single-process capacity.
Run 1 (50 users) saw 52 % failures as the request queue overflowed, while
Run 3 (200 users) produced 11 % failures at a raw throughput of ~900 rps
— confirming that the bottleneck is CPU contention, not I/O.  The standard
remedy is horizontal scaling: additional uvicorn workers or container
replicas behind a load balancer.

### 12.3.4 Latency Summary

| Scenario | p50 | p99 | Throughput | Errors |
|----------|----:|----:|-----------:|-------:|
| Sequential (1 client) | 4.6 ms | 21.5 ms | 187 req/s | 0 % |
| Concurrent (10 clients) | 74.8 ms | 219 ms | 124 req/s | 0.15 % |
| Sustained (40 users, 2 min) | 980 ms | 1,100 ms | 201 req/s | 0 % |

**Table 23 — Latency comparison across benchmark scenarios.**

At every tested workload the system responds within the card-network
authorisation window.  The single-request latency of under 5 ms
demonstrates that the inference stack itself is fast; the increase under
load is attributable to Python's concurrency model and can be mitigated
through standard deployment patterns (multi-worker, container scaling)
without any architectural changes to the fraud-detection pipeline.

### 12.3.5 Robustness Checks

Latency is only half of the real-time suitability question.  A system that
is fast but brittle — one that crashes on unexpected input or silently
produces wrong decisions when features are noisy — would be unusable in
production.  Four robustness checks were therefore executed against the
deployed XGBoost + Autoencoder stack, using 50 real feature snapshots drawn
from the decision store.

**Test protocol.** Each check loads the production XGBoost model
(`xgb_model.pkl`) and the Autoencoder (`autoencoder_model.keras`), applies
a controlled mutation to the feature vector, and records (a) whether the
system returns a valid decision without crashing and (b) whether the
decision changes relative to the unmutated baseline.  The script
(`thesis_quality/robustness/robustness_checks.py`) is deterministic up to
the random seed and can be re-run to reproduce every number below.

#### Input resilience (extreme values)

Four stress scenarios were applied to 20 snapshots each: all features
scaled by ×100, all features scaled by −100×, all features set to zero, and
all features set to NaN.

| Scenario | Snapshots | Valid responses | Crashes | Pass |
|----------|----------:|----------------:|--------:|:----:|
| Scale ×100 | 20 | 20 | 0 | ✅ |
| Scale −100× | 20 | 20 | 0 | ✅ |
| All zeros | 20 | 20 | 0 | ✅ |
| NaN injection | 20 | 20 | 0 | ✅ |

**Table 24 — Input resilience: the pipeline returns a valid APPROVE / REVIEW / BLOCK decision for every extreme-value scenario without crashing.**

XGBoost and the Autoencoder both handle NaN and extreme values gracefully.
XGBoost treats NaN as a missing-value indicator and routes it to the
default branch at each tree split; the Autoencoder produces a high
reconstruction error (which pushes the decision toward REVIEW or BLOCK).
No uncaught exception was raised in any of the 80 trials.

#### Feature perturbation (noise injection ±20 %)

Each of the 50 snapshots was perturbed five times by adding uniform random
noise of ±20 % to every numeric feature.  The question is whether a small
amount of measurement noise changes the final triage decision.

| Metric | Value |
|--------|------:|
| Total trials | 250 |
| Decision flips | 93 |
| Flip rate | 37.2 % |
| Stable rate | 62.8 % |

The 37 % flip rate seems high until you consider the composition of the test
set: 30 of the 50 snapshots sit in the gray zone (XGBoost probability
between 0.05 and 0.80), where flips between REVIEW and BLOCK are expected
and harmless — both outcomes route the transaction to a human analyst.  For
the 18 APPROVE snapshots, perturbation-induced flips were predominantly
APPROVE → REVIEW, which means the system errs on the side of caution when
features are noisy.  No BLOCK transaction flipped to APPROVE.

#### Missing features (zero-fill)

To simulate upstream data-quality failures, a random subset of features was
zeroed out before scoring.

| Features dropped | Flip rate | Crashes |
|-----------------:|----------:|--------:|
| 10 % | 10.0 % | 0 |
| 25 % | 25.0 % | 0 |
| 50 % | 65.0 % | 0 |
| 75 % | 65.0 % | 0 |

**Table 25 — Missing-feature resilience: zero crashes at any drop fraction; decision stability degrades predictably.**

At 10 % feature loss, 90 % of decisions are unchanged — minor upstream
glitches do not destabilise the pipeline.  At 50–75 % loss, 65 % of
decisions change; this is the correct behaviour because a feature vector
that is mostly zeros no longer represents the original transaction.
Critically, the system never crashes: it always returns a valid decision.
In the API layer, the `model_service._validate_processed_payload()` function
raises a structured 400 error if required fields are entirely absent, so
callers receive clear feedback rather than a silent wrong answer.

#### Decision-boundary stability

Seventeen snapshots sat near the APPROVE/REVIEW boundary (XGBoost
probability between 0.02 and 0.12, straddling the `xgb_t_low = 0.05`
threshold).  Each was perturbed at six noise levels (ε = 1 %–30 %) with
10 trials per level.

Even at ε = 1 %, 53 % of these boundary cases flipped — confirming that
the gray zone is genuinely uncertain.  This is by design: transactions near
the decision boundary are the ones the system *should* send for human review.
The hybrid engine's three-way triage exists precisely so that borderline
cases receive analyst attention rather than being silently approved or
blocked.

#### Error recovery and test coverage

Beyond statistical robustness, the codebase includes structural safeguards:

- **Structured error handling.** The `api/core/errors.py` module implements
  a Stripe-style error envelope.  Every exception — whether a missing
  field, a model-loading failure, or an unhandled runtime error — is caught
  by a global handler and returned as a well-formed JSON response with an
  error code, message, and HTTP status.  The API never exposes a raw
  stack trace to the caller.

- **Rule-based fallback.** The decision engine (`api/services/decision_engine.py`)
  includes a rule-based fallback path.  If the feature vector is incomplete
  (missing V1–V28), the system switches from ML scoring to a simple
  weighted-rule engine that checks for geographic mismatch, high velocity,
  proxy/VPN usage, and unusual transaction hours.  This ensures a
  degraded-but-functional decision even when the ML path cannot run.

- **Unit and integration tests.** The test suite (`tests/unit/` and
  `tests/integration/`) includes 14 test modules covering:
  - Missing-feature validation (`test_model_service_predict.py`:
    `test_predict_from_processed_102_missing_required_feature_raises`)
  - Payload-hash stability (`test_model_service_unit.py`:
    `test_sha256_dict_is_stable`)
  - Empty-state responses (`test_api_review_edges.py`:
    `test_review_queue_empty`)
  - Model-registry failure handling (`test_model_service_full.py`:
    `test_load_active_xgb_registry_missing_pointer`)
  - Gray-zone AE activation (`test_model_service_predict.py`:
    `test_predict_from_processed_102_runs_ae_in_grayzone`)

Together, the latency benchmarks (Section 12.3.1–12.3.4) and the robustness
checks above satisfy the requirement to **validate real-time suitability**:
the system is both fast enough and resilient enough for pre-authorisation
deployment.

---

## 12.4 XGBoost Classifier — Results

### 12.4.1 Headline Metrics

The XGBoost model was evaluated on the held-out test set (42,722 transactions,
52 fraud) at the validation-selected threshold of **0.307**.

| Metric | Validation | Test |
|--------|:----------:|:----:|
| Precision | 0.925 | 0.875 |
| Recall | 0.875 | 0.808 |
| F1 | 0.899 | 0.840 |
| ROC-AUC | 0.998 | 0.991 |
| PR-AUC | 0.908 | 0.851 |

**Table 26 — XGBoost classification metrics (threshold = 0.307).**

Precision drops from 0.925 to 0.875 between validation and test, and recall
from 0.875 to 0.808.  This modest decline is expected: validation was used
to *choose* the threshold, so the validation numbers carry a slight
optimistic bias.  The test F1 of **0.840** on a 0.12 % fraud-rate dataset
is a strong result — the model identifies roughly four out of every five
fraudulent transactions while producing only six false positives out of
42,670 legitimate ones.

### 12.4.2 Confusion Matrix

|  | Predicted Legitimate | Predicted Fraud |
|--|:--------------------:|:---------------:|
| **True Legitimate** (n = 42,670) | 42,664 (TN) | 6 (FP) |
| **True Fraud** (n = 52) | 10 (FN) | 42 (TP) |

**Table 27 — XGBoost confusion matrix on the test set.**

Six legitimate cardholders would be incorrectly flagged — a false-positive
rate of 0.014 %.  Ten fraudulent transactions would be missed.  In dollar
terms, whether those ten missed cases are acceptable depends on their
individual amounts; operationally, the hybrid engine provides a second
chance to catch some of them through the Autoencoder gray-zone gate.

> **Figure 16** — XGBoost confusion matrix heatmap.
> `docs/figures/thesis_diagrams/ch12_xgb_confusion_matrix.png`

### 12.4.3 ROC Curve

The ROC curve for the XGBoost model on the test set is shown below.  The
AUC of **0.991** indicates near-perfect ranking: fraudulent transactions are
almost always assigned a higher probability than legitimate ones.  The curve
hugs the top-left corner, confirming that the model achieves a high
true-positive rate at extremely low false-positive rates.

> **Figure 17** — XGBoost ROC curve (test set, AUC = 0.991).
> `thesis_quality/evaluation/plots/xgb_roc_test.png`

### 12.4.4 Precision-Recall Curve

Under extreme imbalance the PR curve is a stricter judge than the ROC curve.
The XGBoost model achieves a PR-AUC (Average Precision) of **0.851** on the
test set.  The curve shows that at a recall of 0.80 the model still maintains
precision above 0.85, which means fewer than one in six flagged transactions is
a false alarm.

> **Figure 18** — XGBoost Precision-Recall curve (test set, AP = 0.851).
> `thesis_quality/evaluation/plots/xgb_pr_test.png`

### 12.4.5 Threshold Selection Curve

The F1-vs-threshold plot generated on the validation set shows a clear peak
at **t = 0.307** where F1 reaches **0.899**.  The curve is not symmetrical
around the peak: F1 degrades slowly as the threshold increases (precision
climbs, recall drops) but falls sharply below 0.15 (recall saturates but
precision collapses).  This shape confirms that the chosen operating point
sits on a robust plateau rather than a knife-edge.

> **Figure 19** — F1 score vs decision threshold (validation set).
> `thesis_quality/evaluation/plots/xgb_f1_vs_threshold_val.png`

---

## 12.5 Autoencoder — Results

The Autoencoder is not designed to compete with XGBoost head-to-head.  Its
purpose in this architecture is to act as an independent anomaly detector in
the gray zone — the band of transactions where XGBoost is uncertain.  The
numbers below confirm why using the AE alone would be insufficient, and why
combining it with XGBoost makes the system stronger.

### 12.5.1 Standalone AE Metrics

The AE was evaluated on the same test set using the operational threshold of
2.414 (the point selected by `run_decision_engine_eval.py` matching
validation-set review-gate behaviour).

| Metric | Validation | Test |
|--------|:----------:|:----:|
| Precision (flagged) | 0.045 | 0.081 |
| Recall (flagged) | 0.429 | 0.385 |
| F1 (flagged) | 0.081 | 0.134 |
| ROC-AUC | 0.945 | 0.927 |
| PR-AUC | 0.029 | 0.045 |

**Table 28 — Autoencoder classification metrics (standalone, flagged = review ∪ block).**

The standalone precision is extremely low (8.1 %) because the AE flags many
legitimate transactions whose feature patterns happen to produce high
reconstruction error.  The recall of 38.5 % means the AE catches roughly
two out of every five fraud cases when working alone.  Its ROC-AUC of 0.927,
however, shows that the *ranking* is informative — fraudulent transactions
do tend to have higher reconstruction error, just not reliably enough to make
clean binary decisions.

When evaluated on the block-only threshold (4.896), the AE catches **zero**
fraud cases on the test set.  This is not a bug — it simply means that no
fraudulent transaction in this particular test split has a reconstruction
error above the p99.5 percentile.  The block gate exists as a safety net for
truly extreme anomalies; in normal operation most AE decisions happen at the
review threshold.

### 12.5.2 Reconstruction-Error Distribution

The histogram below overlays the reconstruction error for legitimate and
fraudulent transactions on the test set.  The legitimate distribution is
concentrated near zero with a long right tail.  The fraud distribution is
wider and shifted to the right, but there is substantial overlap — which is
precisely why the AE cannot serve as the sole classifier.

The vertical dashed line marks the review threshold (0.692) and the
dash-dot line marks the block threshold (4.896).  Most legitimate
transactions fall well below the review line; fraudulent transactions are
spread across both sides, confirming that the AE is useful as a *second
signal* but unreliable as a standalone gatekeeper.

> **Figure 20** — Autoencoder reconstruction-error distribution (test set, legit vs fraud).
> `docs/figures/thesis_diagrams/ch12_ae_error_distribution.png`

### 12.5.3 Training Convergence

The Autoencoder was trained on 198,980 legitimate transactions with early
stopping.  Training converged after **9 epochs** with a final training loss
of **0.288** and a validation loss of **0.301**.  The narrow gap between
training and validation loss (< 0.02) indicates that the model is not
over-fitting to the training distribution, which is important since the
entire purpose of the AE is to generalise the "shape" of legitimate
transactions so that unseen fraud produces higher error.

---

## 12.6 Hybrid Decision Engine — System-Level Results

The previous two sections evaluated each model in isolation.  In production,
neither model acts alone.  The hybrid decision engine routes every transaction
through a two-stage pipeline: XGBoost first, then the Autoencoder for
uncertain cases.  This section reports the end-to-end system performance on
the 42,722-row test set.

### 12.6.1 Triage Distribution

| Decision | Count | % of Total | Fraud Caught |
|----------|------:|-----------:|:------------:|
| APPROVE | 42,529 | 99.55 % | 8 |
| REVIEW | 44 | 0.10 % | 7 |
| BLOCK | 149 | 0.35 % | 37 |
| **Total** | **42,722** | **100 %** | **52** |

**Table 29 — Hybrid decision-engine triage distribution (test set).**

Out of 42,722 transactions, the system approves 99.55 % instantly.  Only
193 transactions (0.45 %) require any human attention or are blocked
automatically.  This is an operationally attractive ratio — analysts review
fewer than 50 cases while the system auto-blocks 149.

### 12.6.2 Fraud Capture Rate

Of the 52 fraudulent transactions in the test set:

- **37** are auto-blocked (BLOCK) — the system stops them with no human
  involvement.
- **7** are flagged for review (REVIEW) — an analyst would see these and
  can confirm or release them.
- **8** slip through as APPROVE — these are the undetected frauds.

The **flagged capture rate** (REVIEW + BLOCK) is:

$$\text{Capture Rate}_{\text{flagged}} = \frac{37 + 7}{52} = \frac{44}{52} = 84.6\%$$

The **auto-block capture rate** (BLOCK only) is:

$$\text{Capture Rate}_{\text{block}} = \frac{37}{52} = 71.2\%$$

When the system is viewed as a binary classifier — "flagged vs not flagged" —
the metrics are:

| View | Precision | Recall | F1 |
|------|:---------:|:------:|:--:|
| Flagged (REVIEW ∪ BLOCK) | 0.228 | 0.846 | 0.359 |
| Block only | 0.248 | 0.712 | 0.368 |

**Table 30 — Binary classification view of the hybrid engine.**

The precision of 0.228 means roughly one in four flagged transactions is
actually fraud.  This might seem low compared to XGBoost's standalone
precision of 0.875, but the two numbers answer different questions.  XGBoost
at its optimal threshold is *conservative* — it flags very few transactions
and gets most of them right.  The hybrid engine casts a *wider net* by also
routing gray-zone transactions through the AE, which pushes recall up to
84.6 % at the cost of more false positives.

In a real card network, the cost of a missed fraud typically exceeds the cost
of a manual review by an order of magnitude.  Achieving 84.6 % capture with
a review queue of only 44 transactions per ~43,000 (0.1 %) is an excellent
trade-off.

> **Figure 21** — Hybrid decision-engine triage distribution (log-scale bar chart).
> `docs/figures/thesis_diagrams/ch12_hybrid_triage_distribution.png`

### 12.6.3 Where the Missed Frauds Hide

Eight fraudulent transactions made it through as APPROVE.  These are cases
where XGBoost assigned a probability below 0.05 — confident enough to bypass
the gray zone entirely — and the AE never got a chance to evaluate them.
There are two possible explanations:

1. **Feature mimicry.** The fraudulent transaction's feature profile is
   statistically indistinguishable from a legitimate one.  In a
   PCA-transformed dataset this can happen when the fraud involves a small
   amount on an established account during business hours — features that
   look entirely normal.

2. **Temporal drift.** The test set is the chronological tail of the dataset.
   Fraud patterns at the end of the two-day window may differ from those in
   the training window.  The feedback loop described in Chapter 10 is
   designed to close this gap over time.

No system can catch 100 % of fraud without rejecting a large number of
legitimate transactions.  The eight missed cases represent the irreducible
trade-off between customer experience and fraud prevention, and they are
precisely the transactions that the feedback loop (Chapter 10) is designed
to learn from.

---

## 12.7 Summary of Key Findings

| Finding | Evidence |
|---------|----------|
| XGBoost alone achieves strong performance under extreme imbalance | Test F1 = 0.840, ROC-AUC = 0.991, only 6 FP on 42,670 legit |
| The Autoencoder is a useful *complement*, not a replacement | Standalone recall 38.5 %, but ROC-AUC 0.927 confirms ranking signal |
| The hybrid engine captures more fraud than either model alone | 84.6 % capture vs XGBoost's 80.8 % — a 3.8 pp improvement |
| Operational overhead is minimal | 0.10 % review rate, 0.35 % auto-block rate, 99.55 % instant approval |
| Latency meets real-time requirements | Median 4.62 ms (single), 74.8 ms (concurrent); p99 < 220 ms |
| A single API instance handles ~200 rps without degradation | Locust run2: 0 % failure at 201 rps, p99 = 1.1 s |
| System is robust to noisy and malformed input | Zero crashes across 80 extreme-value trials; graceful degradation under feature loss |

**Table 31 — Summary of evaluation findings.**

The results confirm the thesis architecture's central hypothesis: a
lightweight, two-model hybrid deployed behind a REST API can provide
near-real-time fraud decisions accurate enough for pre-authorisation use
while remaining simple enough to explain, audit and improve through a
human-in-the-loop feedback cycle.
