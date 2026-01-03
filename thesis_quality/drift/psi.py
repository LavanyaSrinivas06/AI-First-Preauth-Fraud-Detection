# thesis_quality/drift/psi.py
from __future__ import annotations

import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "thesis_quality" / "drift" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

BASELINE = REPO / "artifacts" / "drift" / "baseline_stats.json"
CURRENT = REPO / "data" / "processed" / "test.csv"
FEATURES = REPO / "artifacts" / "preprocess" / "features.json"

GEN_BASELINE = REPO / "scripts" / "drift" / "generate_baseline_stats.py"
DRIFT_MONITOR = REPO / "scripts" / "drift" / "drift_monitor.py"
PSI_ONLY = REPO / "scripts" / "drift" / "psi.py"

STABLE_MAX = 0.10
MODERATE_MAX = 0.20


def run(cmd: list[str]) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def read_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def status_from_max(psi_max: float) -> str:
    if psi_max <= STABLE_MAX:
        return "stable"
    if psi_max <= MODERATE_MAX:
        return "moderate"
    return "high_drift"


def extract_psi_map(report: Dict[str, Any]) -> Dict[str, float]:
    if isinstance(report.get("psi_by_feature"), dict):
        return {k: float(v) for k, v in report["psi_by_feature"].items()}
    if isinstance(report.get("psi"), dict):
        return {k: float(v) for k, v in report["psi"].items()}
    if isinstance(report.get("features"), list):
        out = {}
        for r in report["features"]:
            if isinstance(r, dict) and "name" in r and "psi" in r:
                out[str(r["name"])] = float(r["psi"])
        return out
    return {}


def main() -> None:
    out_json = RESULTS / "psi_report.json"
    out_csv = RESULTS / "psi_report.csv"

    # 1) baseline
    if not BASELINE.exists():
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        run(
            [
                sys.executable,
                str(GEN_BASELINE),
                "--csv-path",
                str(CURRENT),
                "--features-path",
                str(FEATURES),
                "--out-path",
                str(BASELINE),
            ]
        )

    # 2) compute PSI report
    driver = DRIFT_MONITOR if DRIFT_MONITOR.exists() else PSI_ONLY
    if not driver.exists():
        raise FileNotFoundError("No drift driver found under scripts/drift/")

    # try common flags
    cmd = [
        sys.executable,
        str(driver),
        "--baseline-path",
        str(BASELINE),
        "--current-csv",
        str(CURRENT),
        "--features-path",
        str(FEATURES),
        "--out-json",
        str(out_json),
    ]
    try:
        run(cmd)
    except subprocess.CalledProcessError:
        cmd2 = [
            sys.executable,
            str(driver),
            "--baseline",
            str(BASELINE),
            "--current",
            str(CURRENT),
            "--features",
            str(FEATURES),
            "--out",
            str(out_json),
        ]
        run(cmd2)

    report = read_json(out_json)
    psi_map = extract_psi_map(report)
    psi_vals = list(psi_map.values()) if psi_map else [0.0]
    psi_max = max(psi_vals)
    psi_mean = sum(psi_vals) / max(1, len(psi_vals))
    status = status_from_max(psi_max)

    # 3) csv
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["feature", "psi"])
        for k, v in sorted(psi_map.items(), key=lambda x: x[1], reverse=True):
            w.writerow([k, f"{v:.6f}"])

    # 4) readme
    (RESULTS / "README.md").write_text(
        "\n".join(
            [
                "# Drift (PSI) — Thesis Quality",
                "",
                f"- Timestamp (UTC): {datetime.utcnow().isoformat()}Z",
                f"- Status: **{status}**",
                f"- PSI max: **{psi_max:.4f}**",
                f"- PSI mean: **{psi_mean:.4f}**",
                "",
                "Thresholds:",
                f"- stable: max PSI ≤ {STABLE_MAX}",
                f"- moderate: {STABLE_MAX} < max PSI ≤ {MODERATE_MAX}",
                f"- high_drift: max PSI > {MODERATE_MAX}",
                "",
                "Outputs:",
                "- psi_report.json",
                "- psi_report.csv",
            ]
        ),
        encoding="utf-8",
    )

    print(f"[OK] PSI report written to: {RESULTS}")
    print(f"[STATUS] {status} (max={psi_max:.4f}, mean={psi_mean:.4f})")


if __name__ == "__main__":
    main()
