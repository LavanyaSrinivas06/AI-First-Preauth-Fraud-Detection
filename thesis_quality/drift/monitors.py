from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd

from .psi import psi_report, PSIRules


@dataclass
class DriftConfig:
    baseline_csv: Path
    current_csv: Optional[Path] = None
    results_dir: Path = Path("thesis_quality/drift/results")
    label_col: str = "Class"
    n_bins: int = 10
    top_k: int = 20
    rules: PSIRules = PSIRules()


def _load_features(csv_path: Path, label_col: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if label_col in df.columns:
        df = df.drop(columns=[label_col])
    return df


def _classify_status(psi_val: float, rules: PSIRules) -> str:
    if psi_val != psi_val:  # NaN
        return "unknown"
    if psi_val < rules.stable:
        return "stable"
    if psi_val < rules.moderate:
        return "moderate"
    return "significant"


def run_drift_monitor(cfg: DriftConfig) -> Dict[str, Any]:
    cfg.results_dir.mkdir(parents=True, exist_ok=True)

    baseline = _load_features(cfg.baseline_csv, cfg.label_col)
    if cfg.current_csv is None:
        raise ValueError("current_csv is required for this monitor run.")
    current = _load_features(cfg.current_csv, cfg.label_col)

    report_df = psi_report(baseline, current, n_bins=cfg.n_bins)

    # add status labels
    report_df["status"] = report_df["psi"].apply(lambda x: _classify_status(float(x), cfg.rules))

    top = report_df.head(cfg.top_k)

    summary = {
        "baseline": str(cfg.baseline_csv),
        "current": str(cfg.current_csv),
        "n_features_compared": int(report_df.shape[0]),
        "top_k": int(cfg.top_k),
        "top_drifting": top.to_dict(orient="records"),
        "counts_by_status": report_df["status"].value_counts().to_dict(),
    }

    # write JSON
    json_path = cfg.results_dir / "drift_report.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # write MD
    md_path = cfg.results_dir / "drift_report.md"
    md_lines: List[str] = []
    md_lines.append("# Drift Report (PSI)\n")
    md_lines.append(f"- Baseline: `{cfg.baseline_csv}`\n")
    md_lines.append(f"- Current: `{cfg.current_csv}`\n")
    md_lines.append(f"- Features compared: `{summary['n_features_compared']}`\n")
    md_lines.append("\n## Status counts\n")
    for k, v in summary["counts_by_status"].items():
        md_lines.append(f"- {k}: {v}\n")

    md_lines.append("\n## Top drifting features\n")
    md_lines.append("| feature | psi | status |\n")
    md_lines.append("|---|---:|---|\n")
    for r in summary["top_drifting"]:
        md_lines.append(f"| {r['feature']} | {r['psi']:.6f} | {r['status']} |\n")

    md_path.write_text("".join(md_lines), encoding="utf-8")

    # console mini-summary
    print("\n=== Drift Monitor Summary ===")
    print("baseline:", cfg.baseline_csv)
    print("current :", cfg.current_csv)
    print("features:", summary["n_features_compared"])
    print("status_counts:", summary["counts_by_status"])
    print("\nTop drifting:")
    print(top[["feature", "psi", "status"]].to_string(index=False))

    return summary
