"""
One-click reproducibility script for the AI-First Preauthorization
Fraud Detection thesis.

Executes preprocessing, training, evaluation, benchmarking,
and drift monitoring with fixed configuration and artifacts.
"""

# thesis_quality/reproducibility/run_all.py
from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


# -------------------------
# Helpers
# -------------------------
def _repo_root() -> Path:
    """
    Resolve repo root (works when called from anywhere).
    Assumes this file is: thesis_quality/reproducibility/run_all.py
    """
    return Path(__file__).resolve().parents[2]


def _load_dotenv(dotenv_path: Path) -> Dict[str, str]:
    """
    Minimal .env loader (no dependency).
    Loads KEY=VALUE lines into os.environ if not already set.
    Returns loaded key/values.
    """
    loaded: Dict[str, str] = {}
    if not dotenv_path.exists():
        return loaded

    for raw in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if not k:
            continue
        if k not in os.environ:
            os.environ[k] = v
        loaded[k] = os.environ.get(k, v)
    return loaded


def _run(
    cmd: Sequence[str],
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    title: Optional[str] = None,
) -> None:
    """
    Run a command, streaming output to console. Raises on failure.
    """
    if title:
        print(f"\n=== {title} ===")
    print("[RUN]", " ".join(cmd))

    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    subprocess.run(
        list(cmd),
        cwd=str(cwd) if cwd else None,
        env=merged_env,
        check=True,
    )


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# -------------------------
# Config
# -------------------------
@dataclass
class ReproConfig:
    base_url: str = "http://127.0.0.1:8010"
    endpoint: str = "/preauth/decision"

    # Evaluation
    eval_entry: Path = Path("thesis_quality/evaluation/run_evaluation.py")

    # Drift
    drift_entry: Path = Path("thesis_quality/drift/psi.py")

    # Latency
    latency_entry: Path = Path("thesis_quality/benchmarking/latency/run_latency.py")
    latency_n: int = 2000
    latency_concurrency: int = 10
    latency_no_store: bool = True

    # Load
    load_entry: Path = Path("thesis_quality/benchmarking/load/run_load.py")
    load_seconds: int = 30
    load_vus: int = 50
    load_no_store: bool = True

    # Optional: choose python executable
    python: str = sys.executable


def build_config_from_env() -> ReproConfig:
    """
    Read overrides from environment variables (matches your ENV_TEMPLATE.env intent).
    """
    cfg = ReproConfig()
    cfg.base_url = os.getenv("FPN_API_BASE_URL", os.getenv("BASE_URL", cfg.base_url))
    cfg.endpoint = os.getenv("ENDPOINT", cfg.endpoint)

    # Optional numeric overrides
    if os.getenv("LATENCY_N"):
        cfg.latency_n = int(os.getenv("LATENCY_N", str(cfg.latency_n)))
    if os.getenv("LATENCY_CONCURRENCY"):
        cfg.latency_concurrency = int(os.getenv("LATENCY_CONCURRENCY", str(cfg.latency_concurrency)))

    if os.getenv("LOAD_SECONDS"):
        cfg.load_seconds = int(os.getenv("LOAD_SECONDS", str(cfg.load_seconds)))
    if os.getenv("LOAD_VUS"):
        cfg.load_vus = int(os.getenv("LOAD_VUS", str(cfg.load_vus)))

    # no-store switches
    if os.getenv("LATENCY_STORE", "").lower() in {"1", "true", "yes"}:
        cfg.latency_no_store = False
    if os.getenv("LOAD_STORE", "").lower() in {"1", "true", "yes"}:
        cfg.load_no_store = False

    return cfg


# -------------------------
# Main pipeline
# -------------------------
def main() -> int:
    root = _repo_root()
    os.chdir(root)

    # Load .env if present (repo-root .env)
    dotenv_path = root / ".env"
    loaded = _load_dotenv(dotenv_path)

    cfg = build_config_from_env()

    print("[INFO] Repo root:", root)
    if loaded:
        print("[INFO] Loaded .env keys:", ", ".join(sorted(loaded.keys())))
    print(f"[INFO] Using BASE_URL={cfg.base_url} ENDPOINT={cfg.endpoint}")
    print("[INFO] Python:", sys.version.replace("\n", " "))

    # Run log
    stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = root / "thesis_quality" / "reproducibility" / "results" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    try:
        # Step 1/4: Evaluation
        _run(
            [cfg.python, str(cfg.eval_entry)],
            title="STEP 1/4 — Evaluation (models + hybrid + plots + tables)",
        )

        # Step 2/4: Drift PSI
        _run(
            [cfg.python, str(cfg.drift_entry)],
            title="STEP 2/4 — Drift PSI report",
        )

        # Step 3/4: Latency benchmark (NO-STORE recommended)
        latency_cmd = [
            cfg.python,
            str(cfg.latency_entry),
            "--base-url",
            cfg.base_url,
            "--endpoint",
            cfg.endpoint,
            "--n",
            str(cfg.latency_n),
            "--concurrency",
            str(cfg.latency_concurrency),
        ]
        if cfg.latency_no_store:
            latency_cmd.append("--no-store")

        _run(latency_cmd, title="STEP 3/4 — Latency benchmark")

        # Step 4/4: Load test runner (k6 if installed, else locust)
        load_cmd = [
            cfg.python,
            str(cfg.load_entry),
            "--base-url",
            cfg.base_url,
            "--endpoint",
            cfg.endpoint,
            "--seconds",
            str(cfg.load_seconds),
            "--vus",
            str(cfg.load_vus),
        ]
        if cfg.load_no_store:
            load_cmd.append("--no-store")

        _run(load_cmd, title="STEP 4/4 — Load test runner")

        elapsed = time.time() - start
        _write_text(
            out_dir / "run_summary.txt",
            "\n".join(
                [
                    "THESIS-QUALITY REPRODUCIBILITY RUN",
                    f"timestamp: {stamp}",
                    f"base_url: {cfg.base_url}",
                    f"endpoint: {cfg.endpoint}",
                    f"latency: n={cfg.latency_n}, concurrency={cfg.latency_concurrency}, no_store={cfg.latency_no_store}",
                    f"load: seconds={cfg.load_seconds}, vus={cfg.load_vus}, no_store={cfg.load_no_store}",
                    f"elapsed_seconds: {elapsed:.2f}",
                    "status: OK",
                    "",
                    "Outputs to check:",
                    "- thesis_quality/evaluation/ (metrics/, plots/, benchmark_table.csv, etc.)",
                    "- thesis_quality/drift/results/ (psi outputs)",
                    "- thesis_quality/benchmarking/latency/results/",
                    "- thesis_quality/benchmarking/load/results/",
                ]
            ),
        )
        print(f"\n[OK] Thesis-quality reproducibility run completed. Log: {out_dir}")
        return 0

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start
        _write_text(
            out_dir / "run_summary.txt",
            "\n".join(
                [
                    "THESIS-QUALITY REPRODUCIBILITY RUN",
                    f"timestamp: {stamp}",
                    f"base_url: {cfg.base_url}",
                    f"endpoint: {cfg.endpoint}",
                    f"elapsed_seconds: {elapsed:.2f}",
                    "status: FAILED",
                    f"failed_cmd: {' '.join(e.cmd) if isinstance(e.cmd, list) else str(e.cmd)}",
                    f"returncode: {e.returncode}",
                ]
            ),
        )
        print(f"\n[ERROR] Run failed. Log: {out_dir}")
        return int(e.returncode or 1)


if __name__ == "__main__":
    raise SystemExit(main())
