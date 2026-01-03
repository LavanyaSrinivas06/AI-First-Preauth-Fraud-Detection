#!/usr/bin/env python3
"""
Latency + stress benchmark for /preauth/decision.

Reworked to:
- Read processed feature names from artifacts/preprocess/features.json (key: feature_names_after_preprocessing)
- Read processed rows from data/processed/test.csv (default) using pandas
- Provide CLI flags: --n, --concurrency, --endpoint, --base-url, --timeout, --csv-path, --features-path, --seed, --warmup, --no-store
- Warm up with warmup requests, then run N requests with concurrency C
- Ensure unique transaction_id per request (UUID)
- Optionally set header X-No-Store: 1 when --no-store is passed
- Fail fast if /health is unreachable
- Writes artifacts/perf/latency_summary.json and .md
- Uses stdlib + numpy + pandas + httpx
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENDPOINT = "/preauth/decision"
DEFAULT_OUT_DIR = ROOT / "artifacts" / "perf"
DEFAULT_CSV = ROOT / "data" / "processed" / "test.csv"
DEFAULT_FEATURES = ROOT / "artifacts" / "preprocess" / "features.json"
DEFAULT_BASE_URL = "http://127.0.0.1:8010"


def _now_utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _load_feature_names(features_path: Path) -> List[str]:
    """Load the PREPROCESSED feature names that the API expects."""
    if not features_path.exists():
        raise FileNotFoundError(
            f"Features file not found: {features_path}. "
            "Run preprocessing first to generate this file."
        )
    
    try:
        js = json.loads(features_path.read_text(encoding="utf-8"))
        
        # Debug: print what keys are available
        print(f"[DEBUG] Available keys in features.json: {list(js.keys())}")
        
        # Try multiple possible keys
        names = (
            js.get("feature_names_after_preprocessing") or 
            js.get("processed_feature_names") or
            js.get("feature_names")
        )
        
        if not names:
            raise ValueError(
                f"features.json found but missing expected keys. "
                f"Available keys: {list(js.keys())}. "
                "Expected one of: 'feature_names_after_preprocessing', 'processed_feature_names', 'feature_names'"
            )
        
        if not isinstance(names, list) or not names:
            raise ValueError("Feature names must be a non-empty list")
        
        print(f"[DEBUG] Loaded {len(names)} feature names")
        print(f"[DEBUG] First 5 feature names: {names[:5]}")
        
        return list(names)
    except Exception as e:
        raise RuntimeError(f"Failed to read features file {features_path}: {e}")


def _prepare_rows(csv_path: Path, feature_names: List[str], seed: Optional[int]) -> List[Dict[str, Any]]:
    """
    Load CSV and create rows with the exact feature names the API expects.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Processed CSV not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    print(f"[DEBUG] CSV columns ({len(df.columns)}): {list(df.columns)[:10]}...")

    # Drop common label columns if present
    for c in ("Class", "label", "target", "y"):
        if c in df.columns:
            df = df.drop(columns=c)
            print(f"[DEBUG] Dropped label column: {c}")

    # Create mapping: for each required feature name, find the matching CSV column
    # Strategy: 
    # 1. Try exact match first
    # 2. If feature name has prefix like "num__", try stripping prefix
    # 3. If still no match, use 0.0 as default
    
    feature_to_csv = {}
    for feat in feature_names:
        if feat in df.columns:
            # Exact match
            feature_to_csv[feat] = feat
        elif "__" in feat:
            # Try without prefix (e.g., "num__V1" -> "V1")
            suffix = feat.split("__", 1)[1]
            if suffix in df.columns:
                feature_to_csv[feat] = suffix
        # If no match found, it will default to 0.0

    print(f"[DEBUG] Successfully mapped {len(feature_to_csv)}/{len(feature_names)} features to CSV columns")
    if len(feature_to_csv) < len(feature_names):
        unmapped = [f for f in feature_names if f not in feature_to_csv]
        print(f"[DEBUG] Unmapped features (will use 0.0): {unmapped[:10]}...")

    # Optionally shuffle rows
    if seed is not None:
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        d: Dict[str, Any] = {}
        for feat_name in feature_names:
            # Get the CSV column name for this feature
            csv_col = feature_to_csv.get(feat_name)
            
            if csv_col:
                try:
                    v = r[csv_col]
                except Exception:
                    v = 0.0
            else:
                # Feature not in CSV, use default
                v = 0.0
            
            # Handle NaN and type conversion
            if pd.isna(v):
                d[feat_name] = 0.0
            else:
                if isinstance(v, (np.integer,)):
                    d[feat_name] = int(v)
                elif isinstance(v, (np.floating,)):
                    d[feat_name] = float(v)
                else:
                    try:
                        d[feat_name] = float(v)
                    except Exception:
                        d[feat_name] = str(v)
        rows.append(d)

    if not rows:
        raise RuntimeError("No rows available after processing CSV")
    
    print(f"[DEBUG] Prepared {len(rows)} rows")
    print(f"[DEBUG] Sample row keys: {list(rows[0].keys())[:10]}...")
    
    return rows


@dataclass
class ErrorSample:
    status: Optional[int]
    body: str
    latency_ms: float


@dataclass
class BenchResult:
    base_url: str
    endpoint: str
    url: str
    n: int
    concurrency: int
    total_seconds: float
    throughput_rps: float
    error_rate: float
    status_counts: Dict[str, int]
    latency_ms_all: Dict[str, float]
    latency_ms_ok_only: Dict[str, float]
    error_samples: List[Dict[str, Any]]
    timestamp_utc: str


async def _single_request(
    client: httpx.AsyncClient,
    url: str,
    data_payload: Dict[str, Any],
    timeout_s: float,
    headers: Dict[str, str],
    debug_first: bool = False,
) -> Tuple[Optional[int], str, float]:
    # API expects "features" key based on testing
    payload = {
        "transaction_id": f"bench_{uuid.uuid4().hex}",
        "features": data_payload
    }
    
    # Debug first request
    if debug_first:
        print(f"[DEBUG] First request payload keys in 'features': {list(data_payload.keys())[:10]}...")
        print(f"[DEBUG] First request payload sample values: {dict(list(data_payload.items())[:3])}")
        print(f"[DEBUG] Full payload structure: {json.dumps({k: '...' if k == 'features' else v for k, v in payload.items()})}")
    
    t0 = time.perf_counter()
    try:
        r = await client.post(url, json=payload, timeout=timeout_s, headers=headers)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        
        if debug_first:
            print(f"[DEBUG] First response status: {r.status_code}")
            print(f"[DEBUG] First response body: {r.text[:500]}...")
        
        return r.status_code, r.text, latency_ms
    except Exception as e:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return None, repr(e), latency_ms


def _percentiles(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {"min": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0, "mean": 0.0}
    arr = np.array(vals, dtype=float)
    return {
        "min": float(np.min(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def _write_outputs(obj: Dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "latency_summary.json"
    md_path = out_dir / "latency_summary.md"
    json_path.write_text(json.dumps(obj, indent=2), encoding="utf-8")

    sc = obj.get("status_counts", {})
    lat_all = obj.get("latency_ms_all", {})
    lat_ok = obj.get("latency_ms_ok_only", {})
    md = []
    md.append("# Latency + Stress Summary")
    md.append(f"- URL: `{obj.get('url')}`")
    md.append(f"- Requests: **{obj.get('n')}**")
    md.append(f"- Concurrency: **{obj.get('concurrency')}**")
    md.append(f"- Total time: **{obj.get('total_seconds'):.4f}s**")
    md.append(f"- Throughput: **{obj.get('throughput_rps'):.2f} rps**")
    md.append(f"- Error rate: **{obj.get('error_rate'):.3f}**")
    md.append("")
    md.append("## Latency (all responses)")
    md.append(f"- min: {lat_all.get('min', 0):.3f} ms")
    md.append(f"- p50: {lat_all.get('p50', 0):.3f} ms")
    md.append(f"- p90: {lat_all.get('p90', 0):.3f} ms")
    md.append(f"- p99: {lat_all.get('p99', 0):.3f} ms")
    md.append(f"- max: {lat_all.get('max', 0):.3f} ms")
    md.append(f"- mean: {lat_all.get('mean', 0):.3f} ms")
    md.append("")
    md.append("## Latency (2xx only)")
    md.append(f"- min: {lat_ok.get('min', 0):.3f} ms")
    md.append(f"- p50: {lat_ok.get('p50', 0):.3f} ms")
    md.append(f"- p90: {lat_ok.get('p90', 0):.3f} ms")
    md.append(f"- p99: {lat_ok.get('p99', 0):.3f} ms")
    md.append(f"- max: {lat_ok.get('max', 0):.3f} ms")
    md.append(f"- mean: {lat_ok.get('mean', 0):.3f} ms")
    md.append("")
    md.append("## Status counts")
    md.append("```json")
    md.append(json.dumps(sc, indent=2))
    md.append("```")
    md.append("")
    if obj.get("error_samples"):
        md.append("## Error samples (first 10)")
        md.append("```json")
        md.append(json.dumps(obj["error_samples"], indent=2)[:8000])
        md.append("```")

    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")


async def run_bench(
    base_url: str,
    endpoint: str,
    n: int,
    concurrency: int,
    timeout_s: float,
    csv_path: Path,
    features_path: Path,
    seed: Optional[int],
    warmup: int,
    no_store: bool,
    out_dir: Path,
) -> BenchResult:
    url = base_url.rstrip("/") + endpoint
    health_url = base_url.rstrip("/") + "/health"

    print(f"[INFO] Loading feature names from: {features_path}")
    print(f"[INFO] Loading CSV data from: {csv_path}")
    
    # Load the PREPROCESSED feature names (with num__ prefix)
    feature_names = _load_feature_names(features_path)
    
    # Load CSV and map to preprocessed names
    rows = _prepare_rows(csv_path, feature_names, seed)

    headers = {"Content-Type": "application/json"}
    if no_store:
        headers["X-No-Store"] = "1"

    # quick connectivity check (fail early)
    async with httpx.AsyncClient() as check_client:
        try:
            await check_client.get(health_url, timeout=2.0)
            print(f"[INFO] Health check passed: {health_url}")
        except Exception as e:
            raise RuntimeError(
                f"API not reachable at {base_url}. Start the service and verify /health. Root error: {repr(e)}"
            )

    # Warmup: do warmup requests sequentially to avoid polluting concurrency metrics
    if warmup > 0:
        print(f"[INFO] Running {warmup} warmup requests...")
        async with httpx.AsyncClient() as client:
            for i in range(warmup):
                sample = random.choice(rows)
                payload = {
                    "transaction_id": f"warm_{uuid.uuid4().hex}",
                    "features": sample
                }
                try:
                    await client.post(url, json=payload, timeout=timeout_s, headers=headers)
                except Exception:
                    # ignore warmup failures
                    pass

    print(f"[INFO] Starting benchmark: {n} requests with concurrency={concurrency}")

    # Prepare metrics
    latencies_all: List[float] = []
    latencies_ok: List[float] = []
    status_counts = {"2xx": 0, "4xx": 0, "5xx": 0, "other": 0, "exceptions": 0}
    error_samples: List[ErrorSample] = []

    sem = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient() as client:
        async def worker(idx: int) -> None:
            async with sem:
                # pick a row (round-robin when seed provided, else random)
                sample = rows[(idx + (seed or 0)) % len(rows)] if seed is not None else random.choice(rows)
                
                # Debug first request only
                debug_first = (idx == 0)
                status, body, latency_ms = await _single_request(
                    client, url, sample, timeout_s, headers, debug_first
                )
                latencies_all.append(latency_ms)

                if status is None:
                    status_counts["exceptions"] += 1
                    if len(error_samples) < 10:
                        error_samples.append(ErrorSample(None, body, latency_ms))
                    return

                if 200 <= status < 300:
                    status_counts["2xx"] += 1
                    latencies_ok.append(latency_ms)
                elif 400 <= status < 500:
                    status_counts["4xx"] += 1
                    if len(error_samples) < 10:
                        error_samples.append(ErrorSample(status, body, latency_ms))
                elif 500 <= status < 600:
                    status_counts["5xx"] += 1
                    if len(error_samples) < 10:
                        error_samples.append(ErrorSample(status, body, latency_ms))
                else:
                    status_counts["other"] += 1
                    if len(error_samples) < 10:
                        error_samples.append(ErrorSample(status, body, latency_ms))

        t0 = time.perf_counter()
        tasks = [asyncio.create_task(worker(i)) for i in range(n)]
        await asyncio.gather(*tasks)
        total_s = time.perf_counter() - t0

    print(f"[INFO] Benchmark complete in {total_s:.2f}s")

    lat_all_sorted = sorted(latencies_all)
    lat_ok_sorted = sorted(latencies_ok)

    latency_summary_all = _percentiles(lat_all_sorted)
    latency_summary_ok = _percentiles(lat_ok_sorted)

    error_rate = 1.0 - (status_counts["2xx"] / max(1, n))
    throughput = n / total_s if total_s > 0 else 0.0

    res = BenchResult(
        base_url=base_url,
        endpoint=endpoint,
        url=url,
        n=n,
        concurrency=concurrency,
        total_seconds=total_s,
        throughput_rps=throughput,
        error_rate=error_rate,
        status_counts=status_counts,
        latency_ms_all=latency_summary_all,
        latency_ms_ok_only=latency_summary_ok,
        error_samples=[asdict(es) for es in error_samples],
        timestamp_utc=_now_utc_iso(),
    )

    # persist outputs
    _write_outputs(asdict(res), out_dir)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--timeout", type=float, default=5.0)
    ap.add_argument("--csv-path", default=str(DEFAULT_CSV))
    ap.add_argument("--features-path", default=str(DEFAULT_FEATURES))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--no-store", action="store_true", help="Set header X-No-Store: 1 to request API skip persistence")
    args = ap.parse_args()

    try:
        res = asyncio.run(
            run_bench(
                base_url=args.base_url,
                endpoint=args.endpoint,
                n=args.n,
                concurrency=args.concurrency,
                timeout_s=args.timeout,
                csv_path=Path(args.csv_path),
                features_path=Path(args.features_path),
                seed=args.seed,
                warmup=args.warmup,
                no_store=args.no_store,
                out_dir=Path(args.out_dir),
            )
        )
        print(json.dumps(asdict(res), indent=2))
        print(f"\nWrote: {Path(args.out_dir) / 'latency_summary.json'} and {Path(args.out_dir) / 'latency_summary.md'}")
    except Exception as e:
        print(f"ERROR: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()