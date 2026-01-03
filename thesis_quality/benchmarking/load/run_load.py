# thesis_quality/benchmarking/load/run_load.py
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Optional plotting (safe if matplotlib is installed)
try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:
    plt = None


RESULTS_DIR = Path("thesis_quality/benchmarking/load/results")


@dataclass
class RunSummary:
    run_name: str
    endpoint: str
    method: str

    # core counts
    requests: int
    failures: int
    failure_rate: float  # 0..1

    # throughput + latency (ms)
    rps: float
    avg_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float

    # derived
    status: str  # healthy/degraded
    notes: List[str]


def _safe_float(x: str, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _safe_int(x: str, default: int = 0) -> int:
    try:
        # sometimes locust writes floats in csv for counts
        return int(float(x))
    except Exception:
        return default


def _read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [dict(r) for r in reader]


def _pick_aggregated_row(stats_rows: List[Dict[str, str]]) -> Optional[Dict[str, str]]:
    """
    Locust stats.csv typically has:
      Type,Name,Request Count,Failure Count,Median Response Time,Average Response Time,...
    We prefer:
      Type == 'Aggregated' OR Name == 'Aggregated'
    If not present, fall back to POST /preauth/decision row.
    """
    if not stats_rows:
        return None

    # 1) aggregated if exists
    for r in stats_rows:
        t = (r.get("Type") or "").strip().lower()
        name = (r.get("Name") or "").strip().lower()
        if t == "aggregated" or name == "aggregated":
            return r

    # 2) common endpoint row
    for r in stats_rows:
        if (r.get("Name") or "").strip() == "/preauth/decision":
            return r

    # 3) first row
    return stats_rows[0]


def _infer_method_and_name(row: Dict[str, str]) -> Tuple[str, str]:
    # Locust "Type" is method for HTTP
    method = (row.get("Type") or "").strip() or "UNKNOWN"
    name = (row.get("Name") or "").strip() or "UNKNOWN"
    return method, name


def _summarize_run(run_name: str) -> RunSummary:
    stats_path = RESULTS_DIR / f"{run_name}_stats.csv"
    failures_path = RESULTS_DIR / f"{run_name}_failures.csv"
    exceptions_path = RESULTS_DIR / f"{run_name}_exceptions.csv"

    stats_rows = _read_csv_dicts(stats_path)
    agg = _pick_aggregated_row(stats_rows)
    if not agg:
        return RunSummary(
            run_name=run_name,
            endpoint="UNKNOWN",
            method="UNKNOWN",
            requests=0,
            failures=0,
            failure_rate=0.0,
            rps=0.0,
            avg_ms=0.0,
            min_ms=0.0,
            max_ms=0.0,
            p50_ms=0.0,
            p90_ms=0.0,
            p95_ms=0.0,
            p99_ms=0.0,
            status="degraded",
            notes=[f"Missing stats file: {stats_path.as_posix()}"],
        )

    method, endpoint = _infer_method_and_name(agg)

    req = _safe_int(agg.get("Request Count", "0"))
    fail = _safe_int(agg.get("Failure Count", "0"))
    failure_rate = (fail / req) if req > 0 else 0.0

    # Different locust versions name columns slightly differently:
    # - "Average Response Time" (ms)
    # - "Median Response Time" (ms)  (p50)
    # - "Min Response Time", "Max Response Time"
    # - "Requests/s"
    avg_ms = _safe_float(agg.get("Average Response Time", "0"))
    min_ms = _safe_float(agg.get("Min Response Time", "0"))
    max_ms = _safe_float(agg.get("Max Response Time", "0"))
    p50_ms = _safe_float(agg.get("Median Response Time", "0"))
    rps = _safe_float(agg.get("Requests/s", "0"))

    # Percentiles sometimes present as:
    # "95%", "90%", "99%"
    p90_ms = _safe_float(agg.get("90%", "0"))
    p95_ms = _safe_float(agg.get("95%", "0"))
    p99_ms = _safe_float(agg.get("99%", "0"))

    notes: List[str] = []

    # Add top failures/exceptions into notes (thesis-friendly)
    failures_rows = _read_csv_dicts(failures_path)
    if failures_rows:
        # failures.csv usually: Method,Name,# failures
        top = sorted(
            failures_rows,
            key=lambda r: _safe_int(r.get("# failures", "0")),
            reverse=True,
        )[:3]
        for r in top:
            n = _safe_int(r.get("# failures", "0"))
            if n > 0:
                notes.append(f"Top failure: {r.get('Method','?')} {r.get('Name','?')} => {n}")

    exceptions_rows = _read_csv_dicts(exceptions_path)
    if exceptions_rows:
        # exceptions.csv usually: Method,Name,Error,Occurrences
        top = sorted(
            exceptions_rows,
            key=lambda r: _safe_int(r.get("Occurrences", "0")),
            reverse=True,
        )[:3]
        for r in top:
            n = _safe_int(r.get("Occurrences", "0"))
            if n > 0:
                err = (r.get("Error") or "").strip()
                # keep it compact
                if len(err) > 160:
                    err = err[:160] + "..."
                notes.append(f"Top exception ({n}): {err}")

    # Health heuristic (you can tune thresholds)
    status = "healthy"
    if failure_rate > 0.01:
        status = "degraded"
        notes.append(f"Failure rate {failure_rate:.2%} > 1% threshold")
    if p95_ms and p95_ms > 2000:
        status = "degraded"
        notes.append(f"p95 {p95_ms:.0f}ms > 2000ms threshold")

    return RunSummary(
        run_name=run_name,
        endpoint=endpoint,
        method=method,
        requests=req,
        failures=fail,
        failure_rate=failure_rate,
        rps=rps,
        avg_ms=avg_ms,
        min_ms=min_ms,
        max_ms=max_ms,
        p50_ms=p50_ms,
        p90_ms=p90_ms,
        p95_ms=p95_ms,
        p99_ms=p99_ms,
        status=status,
        notes=notes,
    )


def _find_runs() -> List[str]:
    # Detect run names from *_stats.csv
    runs: List[str] = []
    for p in RESULTS_DIR.glob("*_stats.csv"):
        name = p.name.replace("_stats.csv", "")
        runs.append(name)
    return sorted(runs)


def _write_summary_csv(summaries: List[RunSummary], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(s) for s in summaries]
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            # notes as a single string column
            r["notes"] = " | ".join(r["notes"])
            w.writerow(r)


def _plot_history(run_name: str) -> None:
    """
    Uses *_stats_history.csv if present.
    Creates throughput_over_time.png and p95_over_time.png per run.
    """
    hist_path = RESULTS_DIR / f"{run_name}_stats_history.csv"
    if not hist_path.exists() or plt is None:
        return

    rows = _read_csv_dicts(hist_path)
    if not rows:
        return

    # stats_history.csv typically includes:
    # Timestamp,User Count,Type,Name,Requests/s,Failures/s,50%,95%,...
    # We focus on aggregated row(s).
    xs: List[int] = []
    rps: List[float] = []
    p95: List[float] = []

    for r in rows:
        t = (r.get("Type") or "").strip().lower()
        name = (r.get("Name") or "").strip().lower()
        if not (t == "aggregated" or name == "aggregated"):
            continue
        ts = _safe_int(r.get("Timestamp", "0"))
        xs.append(ts)
        rps.append(_safe_float(r.get("Requests/s", "0")))
        p95.append(_safe_float(r.get("95%", "0")))

    if not xs:
        return

    # normalize time axis to seconds since start
    t0 = xs[0]
    xsec = [x - t0 for x in xs]

    out_dir = RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # throughput plot
    plt.figure()
    plt.plot(xsec, rps)
    plt.xlabel("Seconds since start")
    plt.ylabel("Requests/sec")
    plt.title(f"{run_name}: Throughput over time")
    plt.tight_layout()
    plt.savefig(out_dir / f"{run_name}_throughput_over_time.png", dpi=160)
    plt.close()

    # p95 plot
    plt.figure()
    plt.plot(xsec, p95)
    plt.xlabel("Seconds since start")
    plt.ylabel("p95 latency (ms)")
    plt.title(f"{run_name}: p95 latency over time")
    plt.tight_layout()
    plt.savefig(out_dir / f"{run_name}_p95_over_time.png", dpi=160)
    plt.close()


def _render_md(summaries: List[RunSummary]) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: List[str] = []
    lines.append("# Load Test Report")
    lines.append("")
    lines.append(f"- Generated: `{now}`")
    lines.append(f"- Runs detected: `{len(summaries)}`")
    lines.append("")
    if not summaries:
        lines.append("No runs found (expected `*_stats.csv`).")
        return "\n".join(lines)

    # Quick table
    lines.append("## Summary")
    lines.append("")
    lines.append("| run | status | requests | failures | failure_rate | rps | p50_ms | p95_ms | p99_ms | max_ms |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s in summaries:
        lines.append(
            f"| {s.run_name} | {s.status} | {s.requests} | {s.failures} | {s.failure_rate:.2%} | "
            f"{s.rps:.2f} | {s.p50_ms:.0f} | {s.p95_ms:.0f} | {s.p99_ms:.0f} | {s.max_ms:.0f} |"
        )

    lines.append("")
    lines.append("## Run details")
    lines.append("")
    for s in summaries:
        lines.append(f"### {s.run_name}")
        lines.append("")
        lines.append(f"- Endpoint: `{s.method} {s.endpoint}`")
        lines.append(f"- Status: **{s.status}**")
        lines.append(f"- Requests: `{s.requests}`  | Failures: `{s.failures}`  | Failure rate: `{s.failure_rate:.2%}`")
        lines.append(f"- Throughput: `{s.rps:.2f} req/s`")
        lines.append(
            f"- Latency (ms): min `{s.min_ms:.0f}` | avg `{s.avg_ms:.0f}` | p50 `{s.p50_ms:.0f}` | "
            f"p90 `{s.p90_ms:.0f}` | p95 `{s.p95_ms:.0f}` | p99 `{s.p99_ms:.0f}` | max `{s.max_ms:.0f}`"
        )

        if s.notes:
            lines.append("- Notes:")
            for n in s.notes:
                lines.append(f"  - {n}")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    runs = _find_runs()
    summaries = [_summarize_run(r) for r in runs]

    # write outputs
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "runs": [asdict(s) for s in summaries],
    }

    (RESULTS_DIR / "load_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (RESULTS_DIR / "load_report.md").write_text(_render_md(summaries), encoding="utf-8")
    _write_summary_csv(summaries, RESULTS_DIR / "load_summary.csv")

    # optional plots
    for r in runs:
        _plot_history(r)

    print(f"[OK] wrote: {RESULTS_DIR / 'load_report.json'}")
    print(f"[OK] wrote: {RESULTS_DIR / 'load_report.md'}")
    print(f"[OK] wrote: {RESULTS_DIR / 'load_summary.csv'}")
    if plt is None:
        print("[NOTE] matplotlib not available; skipped plots.")
    else:
        print("[OK] plots (if history files existed) saved in results/.")


if __name__ == "__main__":
    main()
