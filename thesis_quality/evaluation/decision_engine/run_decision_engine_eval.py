# thesis_quality/evaluation/decision_engine/run_decision_engine_eval.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple

import joblib
import numpy as np
import pandas as pd


@dataclass
class DecisionEngineEvalConfig:
    test_csv: Path = Path("data/processed/test.csv")
    xgb_model_path: Path = Path("artifacts/models/xgb_model.pkl")
    ae_test_errors_path: Path = Path("artifacts/ae_errors/ae_test_errors.npy")

    # These should match your API config/thresholds:
    xgb_t_low: float = 0.05
    xgb_t_high: float = 0.80

    # AE thresholds:
    # - ae_threshold: “review threshold” used for anomaly detection
    # - ae_block: “hard block gate” (you saw ~4.895553 in your API output)
    ae_threshold: float = 2.413630972099965
    ae_block: float = 4.895553

    out_dir: Path = Path("thesis_quality/evaluation/decision_engine")


def decision_engine(p_xgb: float, ae_err: float, cfg: DecisionEngineEvalConfig) -> Tuple[str, Dict[str, Any]]:
    """
    System-level decision logic mirroring your hybrid engine:
    - If AE hard gate triggers -> BLOCK
    - Else XGB high confidence -> BLOCK
    - Else XGB low confidence -> APPROVE
    - Else -> REVIEW (gray zone)
    """
    reasons = []

    # AE hard block gate
    if ae_err is not None and ae_err >= cfg.ae_block:
        reasons.append("ae_block_gate")
        return "BLOCK", {"reasons": reasons}

    # XGB rule
    if p_xgb >= cfg.xgb_t_high:
        reasons.append("xgb_block_high")
        return "BLOCK", {"reasons": reasons}

    if p_xgb <= cfg.xgb_t_low:
        reasons.append("xgb_approve_low")
        return "APPROVE", {"reasons": reasons}

    reasons.append("xgb_gray_zone")
    return "REVIEW", {"reasons": reasons}


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    # y_true, y_pred are 0/1
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def prf_from_counts(c: Dict[str, int]) -> Dict[str, float]:
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def main() -> None:
    cfg = DecisionEngineEvalConfig()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(cfg.test_csv)
    if "Class" not in df.columns:
        raise ValueError("Expected 'Class' column in test.csv")

    y_true = df["Class"].astype(int).to_numpy()
    X = df.drop(columns=["Class"])

    # XGB probs
    xgb = joblib.load(cfg.xgb_model_path)
    p = xgb.predict_proba(X)[:, 1].astype(float)

    # AE errors aligned with test rows
    ae_err = np.load(cfg.ae_test_errors_path).astype(float)
    if len(ae_err) != len(df):
        raise ValueError(f"AE errors length {len(ae_err)} != test rows {len(df)}")

    # Decisions
    decisions = []
    for px, ae in zip(p, ae_err):
        d, meta = decision_engine(px, ae, cfg)
        decisions.append(d)

    decisions = np.array(decisions)

    # --- System binary views ---
    # View A: "flagged" = (REVIEW or BLOCK)
    pred_flagged = np.isin(decisions, ["REVIEW", "BLOCK"]).astype(int)
    c_flagged = confusion_counts(y_true, pred_flagged)
    m_flagged = prf_from_counts(c_flagged)

    # View B: "auto_block" = (BLOCK only)
    pred_block = (decisions == "BLOCK").astype(int)
    c_block = confusion_counts(y_true, pred_block)
    m_block = prf_from_counts(c_block)

    # --- Triage rates ---
    total = len(decisions)
    triage = {
        "total": int(total),
        "approve_n": int((decisions == "APPROVE").sum()),
        "review_n": int((decisions == "REVIEW").sum()),
        "block_n": int((decisions == "BLOCK").sum()),
    }
    triage["approve_rate"] = triage["approve_n"] / total
    triage["review_rate"] = triage["review_n"] / total
    triage["block_rate"] = triage["block_n"] / total

    # Fraud capture inside triage buckets (very thesis-relevant)
    fraud_total = int((y_true == 1).sum())
    triage_fraud = {
        "fraud_total": fraud_total,
        "fraud_in_approve": int(((y_true == 1) & (decisions == "APPROVE")).sum()),
        "fraud_in_review": int(((y_true == 1) & (decisions == "REVIEW")).sum()),
        "fraud_in_block": int(((y_true == 1) & (decisions == "BLOCK")).sum()),
    }
    triage_fraud["capture_rate_flagged"] = (
        1.0 - (triage_fraud["fraud_in_approve"] / fraud_total) if fraud_total > 0 else 0.0
    )
    triage_fraud["capture_rate_block_only"] = (
        (triage_fraud["fraud_in_block"] / fraud_total) if fraud_total > 0 else 0.0
    )

    metrics = {
        "config": {
            "xgb_t_low": cfg.xgb_t_low,
            "xgb_t_high": cfg.xgb_t_high,
            "ae_threshold": cfg.ae_threshold,
            "ae_block": cfg.ae_block,
        },
        "triage": triage,
        "triage_fraud": triage_fraud,
        "binary_flagged_view": {**m_flagged, **c_flagged},
        "binary_block_only_view": {**m_block, **c_block},
    }

    # Save JSON
    (cfg.out_dir / "decision_engine_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    # Save benchmark table CSV (like your model benchmark_table.csv)
    rows = [
        {
            "system_view": "flagged_(review_or_block)",
            **metrics["binary_flagged_view"],
            "approve_rate": triage["approve_rate"],
            "review_rate": triage["review_rate"],
            "block_rate": triage["block_rate"],
        },
        {
            "system_view": "block_only",
            **metrics["binary_block_only_view"],
            "approve_rate": triage["approve_rate"],
            "review_rate": triage["review_rate"],
            "block_rate": triage["block_rate"],
        },
    ]
    pd.DataFrame(rows).to_csv(cfg.out_dir / "decision_engine_benchmark_table.csv", index=False)

    # Save a short report
    report_md = f"""# Decision-Engine Evaluation (System-Level)

Inputs:
- test_csv: `{cfg.test_csv}`
- xgb_model: `{cfg.xgb_model_path}`
- ae_errors: `{cfg.ae_test_errors_path}`

Config:
- xgb_t_low={cfg.xgb_t_low}, xgb_t_high={cfg.xgb_t_high}
- ae_block={cfg.ae_block}

## Triage rates
- APPROVE: {triage["approve_n"]} ({triage["approve_rate"]:.4f})
- REVIEW : {triage["review_n"]} ({triage["review_rate"]:.4f})
- BLOCK  : {triage["block_n"]} ({triage["block_rate"]:.4f})

## Fraud distribution by bucket
- Fraud total: {triage_fraud["fraud_total"]}
- Fraud in APPROVE: {triage_fraud["fraud_in_approve"]}
- Fraud in REVIEW : {triage_fraud["fraud_in_review"]}
- Fraud in BLOCK  : {triage_fraud["fraud_in_block"]}

Capture:
- Flagged capture (REVIEW or BLOCK): {triage_fraud["capture_rate_flagged"]:.4f}
- Auto-block capture (BLOCK only): {triage_fraud["capture_rate_block_only"]:.4f}

## Binary evaluation (Flagged = REVIEW or BLOCK)
- Precision: {m_flagged["precision"]:.4f}
- Recall   : {m_flagged["recall"]:.4f}
- F1       : {m_flagged["f1"]:.4f}
- TN/FP/FN/TP: {c_flagged["tn"]}/{c_flagged["fp"]}/{c_flagged["fn"]}/{c_flagged["tp"]}

## Binary evaluation (Auto-block only)
- Precision: {m_block["precision"]:.4f}
- Recall   : {m_block["recall"]:.4f}
- F1       : {m_block["f1"]:.4f}
- TN/FP/FN/TP: {c_block["tn"]}/{c_block["fp"]}/{c_block["fn"]}/{c_block["tp"]}
"""
    (cfg.out_dir / "decision_engine_report.md").write_text(report_md, encoding="utf-8")

    print("[OK] Wrote:")
    print("-", cfg.out_dir / "decision_engine_metrics.json")
    print("-", cfg.out_dir / "decision_engine_benchmark_table.csv")
    print("-", cfg.out_dir / "decision_engine_report.md")


if __name__ == "__main__":
    main()
