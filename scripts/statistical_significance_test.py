"""
Statistical Significance Test: Hybrid vs XGBoost

Performs McNemar's exact test and bootstrap confidence intervals
for precision, recall, and F1-score differences.

- Loads y_true, XGBoost, and Hybrid predictions from comparative_eval.py pipeline
- No hardcoded paths; auto-discovers test set and artifacts
- Prints results in thesis-ready format
- Reproducible (random seed=42)
"""
import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm

np.random.seed(42)

# === PART 1: Load Predictions ===
# Try to find the comparative_eval results or generate from scratch
PRED_PATHS = [
    Path("artifacts/test_predictions.csv"),
    Path("experiments/comparative_eval_results.csv"),
]

pred_path = next((p for p in PRED_PATHS if p.exists()), None)
if pred_path is None:
    raise FileNotFoundError("No per-transaction prediction file found. Please run comparative_eval.py and ensure predictions are saved.")

# Load predictions
if pred_path.suffix == ".csv":
    df = pd.read_csv(pred_path)
else:
    raise ValueError(f"Unsupported prediction file: {pred_path}")

# Try to find columns
label_col = None
for col in ["y_true", "label", "Class", "target"]:
    if col in df.columns:
        label_col = col
        break
if label_col is None:
    raise ValueError(f"No ground truth column found in {pred_path}. Columns: {list(df.columns)}")

y_true = df[label_col].values.astype(int)

# XGBoost: use probability if available, else binary
xgb_flag = None
for col in ["xgb_flag", "xgb_pred", "xgb_preds", "xgb", "xgb_prediction", "xgb_prob"]:
    if col in df.columns:
        xgb_flag = df[col].values
        break
if xgb_flag is None:
    # Try to reconstruct from probability and threshold
    for col in ["xgb_prob", "xgb_probability", "xgb_probs"]:
        if col in df.columns:
            xgb_flag = (df[col].values >= 0.307).astype(int)
            break
if xgb_flag is None:
    raise ValueError("No XGBoost prediction column found.")

# Hybrid: use flag if available, else reconstruct
hybrid_flag = None
for col in ["hybrid_flag", "hybrid_pred", "hybrid_preds", "hybrid", "hybrid_prediction"]:
    if col in df.columns:
        hybrid_flag = df[col].values
        break
if hybrid_flag is None:
    # Try to reconstruct from decision column
    for col in ["hybrid_decision", "hybrid_decisions"]:
        if col in df.columns:
            # Map BLOCK/REVIEW to 1, APPROVE to 0
            hybrid_flag = np.where(df[col].isin(["BLOCK", "REVIEW", 1, 2]), 1, 0)
            break
if hybrid_flag is None:
    raise ValueError("No Hybrid prediction column found.")

# Ensure all arrays are same length
if not (len(y_true) == len(xgb_flag) == len(hybrid_flag)):
    raise ValueError("Prediction arrays are not aligned.")

# === PART 2: McNemar's Test ===
def mcnemar_test(y_true, xgb_flag, hybrid_flag, alpha=0.05):
    xgb_correct = (xgb_flag == y_true)
    hybrid_correct = (hybrid_flag == y_true)
    n11 = int(np.sum(xgb_correct & hybrid_correct))
    n00 = int(np.sum(~xgb_correct & ~hybrid_correct))
    n01 = int(np.sum(xgb_correct & ~hybrid_correct))
    n10 = int(np.sum(~xgb_correct & hybrid_correct))
    table = np.array([[n00, n10], [n01, n11]])
    discordant = n01 + n10
    exact = True
    result = mcnemar(table, exact=exact)
    stat = result.statistic
    pvalue = result.pvalue
    print("\nMcNemar Exact Test Results")
    print("Contingency Table:")
    print(table)
    print(f"n01 (XGB correct, Hybrid wrong) = {n01}")
    print(f"n10 (XGB wrong, Hybrid correct) = {n10}")
    print(f"p-value = {pvalue:.5f}")
    if pvalue < alpha:
        if n10 > n01:
            print(f"Result: Hybrid is statistically significantly better than XGBoost (p < {alpha})")
        elif n01 > n10:
            print(f"Result: XGBoost is statistically significantly better than Hybrid (p < {alpha})")
        else:
            print(f"Result: Significant difference detected, but discordant counts are equal.")
    else:
        print(f"Result: No statistically significant difference (p >= {alpha})")
    return dict(table=table, n00=n00, n01=n01, n10=n10, n11=n11, stat=stat, pvalue=pvalue)

# === PART 3: Bootstrap Confidence Intervals ===
def bootstrap_ci(y_true, xgb_flag, hybrid_flag, n_boot=5000, alpha=0.05):
    rng = np.random.default_rng(42)
    n = len(y_true)
    deltas = {"precision": [], "recall": [], "f1": []}
    for _ in tqdm(range(n_boot), desc="Bootstrapping"):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        xg = xgb_flag[idx]
        hy = hybrid_flag[idx]
        deltas["precision"].append(precision_score(yt, hy) - precision_score(yt, xg))
        deltas["recall"].append(recall_score(yt, hy) - recall_score(yt, xg))
        deltas["f1"].append(f1_score(yt, hy) - f1_score(yt, xg))
    results = {}
    for k in deltas:
        arr = np.array(deltas[k])
        mean = arr.mean()
        lower = np.percentile(arr, 2.5)
        upper = np.percentile(arr, 97.5)
        results[k] = dict(mean=mean, lower=lower, upper=upper, crosses_zero=(lower <= 0 <= upper))
    print("\nBootstrap Results (95% CI)")
    for k, v in results.items():
        print(f"Δ{k.capitalize()} = {v['mean']:.3f} (CI: {v['lower']:.3f} – {v['upper']:.3f})", end=" ")
        if v['crosses_zero']:
            print("[CI crosses zero]")
        else:
            print("[Significant]")
    return results

if __name__ == "__main__":
    mcnemar_test(y_true, xgb_flag, hybrid_flag)
    bootstrap_ci(y_true, xgb_flag, hybrid_flag)
