# dashboard/app.py

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st
import yaml

from utils import (
    append_review_log,
    compute_amount_threshold,
    find_shap_png,
    find_fallback_shap,
    generate_reasons_for_row,
    load_review_log,
)


# ----------------------------
# Config
# ----------------------------

CONFIG_PATH = Path("dashboard/config.yaml")


@st.cache_data(show_spinner=False)
def load_config() -> Dict[str, Any]:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f) or {}


def cfg_get(cfg: Dict[str, Any], *keys: str, default=None):
    cur = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def col(cfg: Dict[str, Any], group: str, name: str, default: str) -> str:
    return cfg_get(cfg, group, name, default=default)


# ----------------------------
# Data loading
# ----------------------------

def api_get_json(url: str, timeout: float = 5.0) -> Any:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()


def load_queue_api(cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Expected API contract (recommended):
    GET /queue -> {"items":[{...},{...}]}
    Each item should contain:
      txn_id, timestamp, amount, score_xgb, score_ae, ensemble_score, proposed_label (optional),
      reasons: [str,...] (optional),
      top_features: [{"name": "...", "value": ..., "direction": "high|low", "importance": ...}, ...] (optional)
    """
    base = cfg_get(cfg, "data_source", "api_base_url", default="http://127.0.0.1:8000")
    data = api_get_json(f"{base}/queue")
    items = data.get("items", data)  # allow list fallback
    df = pd.DataFrame(items)
    return df


def load_queue_local(cfg: Dict[str, Any], min_score: float, max_rows: int) -> pd.DataFrame:
    path = cfg_get(cfg, "paths", "enriched_parquet", default="data/processed/enriched.parquet")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Local enriched parquet not found: {path} Expected at: {path}")

    df = pd.read_parquet(p)

    ens_col = col(cfg, "scores", "ensemble_score", "ensemble_score")
    if ens_col not in df.columns:
        raise KeyError(f"Missing ensemble score column '{ens_col}' in {path}")

    df = df.sort_values(by=ens_col, ascending=False)
    df = df[df[ens_col] >= float(min_score)].head(int(max_rows)).copy()
    return df


def ensure_reasons_column(cfg: Dict[str, Any], df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[float]]:
    """
    Adds:
      - why_flagged (str) short message
      - reasons (list[str]) if missing (local mode or api fallback)
    """
    if df.empty:
        return df, None

    amount_col = col(cfg, "columns", "amount", "amount")
    q = float(cfg_get(cfg, "explainability", "high_amount_quantile", default=0.95))
    amount_thr = compute_amount_threshold(df, amount_col, q) if amount_col in df.columns else None

    # normalize reasons
    if "reasons" not in df.columns:
        df["reasons"] = None

    # compute missing reasons or convert bad types
    def _norm_reasons(x) -> Optional[List[str]]:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return None
        if isinstance(x, list):
            return [str(s) for s in x if str(s)]
        if isinstance(x, str):
            # allow comma-separated
            parts = [p.strip() for p in x.split(",") if p.strip()]
            return parts if parts else None
        return None

    df["reasons"] = df["reasons"].apply(_norm_reasons)

    # compute reasons if missing
    def _compute_if_none(row: pd.Series) -> List[str]:
        r = row.get("reasons")
        if isinstance(r, list) and len(r) > 0:
            return r
        return generate_reasons_for_row(row, cfg, amount_threshold=amount_thr)

    df["reasons"] = df.apply(_compute_if_none, axis=1)

    # create short message
    def _short_msg(rs: List[str]) -> str:
        if not rs:
            return "Flagged: elevated risk score"
        if len(rs) == 1:
            return rs[0]
        # show top 2
        return f"{rs[0]} • {rs[1]}"

    df["why_flagged"] = df["reasons"].apply(_short_msg)
    return df, amount_thr


def load_queue(cfg: Dict[str, Any], mode: str, min_score: float, max_rows: int) -> pd.DataFrame:
    if mode == "api":
        df = load_queue_api(cfg)
    else:
        df = load_queue_local(cfg, min_score=min_score, max_rows=max_rows)

    df, _ = ensure_reasons_column(cfg, df)
    return df


# ----------------------------
# Page helpers
# ----------------------------

def format_queue(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    if df.empty:
        return df

    txn_id = col(cfg, "columns", "txn_id", "txn_id")
    ts = col(cfg, "columns", "timestamp", "timestamp")
    amt = col(cfg, "columns", "amount", "amount")
    sx = col(cfg, "scores", "score_xgb", "score_xgb")
    sa = col(cfg, "scores", "score_ae", "score_ae")
    se = col(cfg, "scores", "ensemble_score", "ensemble_score")
    pl = col(cfg, "scores", "proposed_label", "proposed_label")

    wanted = [c for c in [txn_id, ts, amt, sx, sa, se, pl, "why_flagged"] if c in df.columns]
    out = df[wanted].copy()
    return out


def set_selected_txn(txn_id: str):
    st.session_state["selected_txn_id"] = txn_id
    st.session_state["page"] = "Details"


def get_selected_txn() -> Optional[str]:
    return st.session_state.get("selected_txn_id")


def sidebar(cfg: Dict[str, Any]) -> Tuple[str, str, float, int]:
    st.sidebar.header("Settings")
    mode_default = cfg_get(cfg, "data_source", "mode", default="local")
    mode = st.sidebar.selectbox("Data source mode", ["local", "api"], index=0 if mode_default == "local" else 1)
    analyst = st.sidebar.text_input("Analyst", value=st.session_state.get("analyst", "analyst@fpn"))
    st.session_state["analyst"] = analyst

    st.sidebar.divider()
    st.sidebar.subheader("Queue filters")
    min_default = float(cfg_get(cfg, "queue", "min_ensemble_score", default=0.58))
    min_score = st.sidebar.slider("Min ensemble_score", 0.0, 1.0, min_default, 0.01)

    max_rows = int(cfg_get(cfg, "queue", "max_rows", default=500))

    st.sidebar.divider()
    st.sidebar.subheader("Pages")
    page = st.sidebar.radio(" ", ["Queue", "Details", "Metrics"], index=["Queue", "Details", "Metrics"].index(st.session_state.get("page", "Queue")))
    st.session_state["page"] = page

    return mode, analyst, min_score, max_rows


# ----------------------------
# Pages
# ----------------------------

def page_queue(cfg: Dict[str, Any], mode: str, min_score: float, max_rows: int):
    st.subheader("Queue")

    try:
        df = load_queue(cfg, mode=mode, min_score=min_score, max_rows=max_rows)
    except Exception as e:
        st.error(f"Queue load failed: {e}")
        st.info("Queue is empty (or filters removed all rows).")
        return

    if df.empty:
        st.info("Queue is empty (or filters removed all rows).")
        return

    # Basic filters (safe if columns missing)
    ts_col = col(cfg, "columns", "timestamp", "timestamp")
    se_col = col(cfg, "scores", "ensemble_score", "ensemble_score")
    country_col = col(cfg, "columns", "country", "country")

    fdf = df.copy()
    with st.expander("Filters", expanded=False):
        if ts_col in fdf.columns:
            # handle string timestamps
            tmp = pd.to_datetime(fdf[ts_col], errors="coerce")
            min_dt = tmp.min()
            max_dt = tmp.max()
            if pd.notna(min_dt) and pd.notna(max_dt):
                d1, d2 = st.date_input("Date range", value=(min_dt.date(), max_dt.date()))
                mask = (tmp.dt.date >= d1) & (tmp.dt.date <= d2)
                fdf = fdf[mask]

        if se_col in fdf.columns:
            lo, hi = st.slider("Score range", 0.0, 1.0, (float(min_score), 1.0), 0.01)
            fdf = fdf[(fdf[se_col] >= lo) & (fdf[se_col] <= hi)]

        if country_col in fdf.columns:
            countries = sorted([c for c in fdf[country_col].dropna().astype(str).unique()])
            if countries:
                pick = st.multiselect("Country", countries, default=[])
                if pick:
                    fdf = fdf[fdf[country_col].astype(str).isin(pick)]

    show = format_queue(fdf, cfg)
    if show.empty:
        st.info("No rows after filters.")
        return

    st.caption("Click a transaction ID to open Details.")
    txn_id_col = col(cfg, "columns", "txn_id", "txn_id")

    # Render clickable table-ish view
    for _, r in show.iterrows():
        cols = st.columns([2.2, 2.2, 1.2, 1.2, 1.2, 1.4, 4.0])
        if txn_id_col in show.columns:
            with cols[0]:
                if st.button(str(r[txn_id_col]), key=f"open_{r[txn_id_col]}"):
                    set_selected_txn(str(r[txn_id_col]))
        with cols[1]:
            st.write(str(r.get(col(cfg, "columns", "timestamp", "timestamp"), "")))
        with cols[2]:
            st.write(r.get(col(cfg, "columns", "amount", "amount"), ""))
        with cols[3]:
            st.write(r.get(col(cfg, "scores", "score_xgb", "score_xgb"), ""))
        with cols[4]:
            st.write(r.get(col(cfg, "scores", "score_ae", "score_ae"), ""))
        with cols[5]:
            st.write(r.get(col(cfg, "scores", "ensemble_score", "ensemble_score"), ""))
        with cols[6]:
            st.write(r.get("why_flagged", ""))


def page_details(cfg: Dict[str, Any], mode: str, analyst: str, min_score: float, max_rows: int):
    st.subheader("Details")

    txn = get_selected_txn()
    if not txn:
        st.info("Select a transaction from Queue first.")
        return

    try:
        df = load_queue(cfg, mode=mode, min_score=min_score, max_rows=max_rows)
    except Exception as e:
        st.error(f"Failed to load queue: {e}")
        return

    txn_id_col = col(cfg, "columns", "txn_id", "txn_id")
    if txn_id_col not in df.columns:
        st.error(f"Queue is missing txn_id column '{txn_id_col}'")
        return

    row_df = df[df[txn_id_col].astype(str) == str(txn)]
    if row_df.empty:
        st.warning("Selected transaction not found in current queue (maybe filters changed).")
        return

    row = row_df.iloc[0]

    # Reasons / abnormal message
    reasons = row.get("reasons") if "reasons" in row_df.columns else []
    if not isinstance(reasons, list):
        reasons = []
    st.markdown("### Why it was flagged")
    if reasons:
        for s in reasons:
            st.write(f"- {s}")
    else:
        st.write("- Flagged due to elevated risk score (no rule-based reasons available).")

    # Scores
    st.markdown("### Scores")
    sx = col(cfg, "scores", "score_xgb", "score_xgb")
    sa = col(cfg, "scores", "score_ae", "score_ae")
    se = col(cfg, "scores", "ensemble_score", "ensemble_score")
    c1, c2, c3 = st.columns(3)
    c1.metric("score_xgb", f"{row.get(sx, '')}")
    c2.metric("score_ae", f"{row.get(sa, '')}")
    c3.metric("ensemble_score", f"{row.get(se, '')}")

    # Raw features
    with st.expander("Raw features", expanded=False):
        st.dataframe(pd.DataFrame(row).rename(columns={0: "value"}), use_container_width=True)

    # SHAP explainability
    st.markdown("### Model explainability (SHAP)")
    static_dir = cfg_get(cfg, "paths", "static_dir", default="dashboard/static")

    shap_png = find_shap_png(static_dir, str(txn))
    fallback_shap = find_fallback_shap(static_dir)

    if shap_png:
        st.image(shap_png, caption=f"SHAP explanation for txn_id={txn}", use_container_width=True)
    elif fallback_shap:
        st.image(fallback_shap, caption="SHAP example (no transaction-specific SHAP found)", use_container_width=True)
    else:
        st.info("No SHAP explanation available for this transaction.")

    # Actions (persist)
    st.markdown("### Analyst action")
    notes = st.text_area("Notes", value="", placeholder="Add optional notes…")
    decision_time = datetime.utcnow().isoformat()

    review_path = cfg_get(cfg, "paths", "review_log_parquet", default="data/review_log.parquet")
    ts_col = col(cfg, "columns", "timestamp", "timestamp")

    def write_decision(decision: str):
        record = {
            "txn_id": str(row.get(txn_id_col)),
            "timestamp": str(row.get(ts_col, "")),
            "ensemble_score": float(row.get(se)) if se in df.columns and pd.notna(row.get(se)) else None,
            "analyst_decision": decision,
            "analyst": analyst,
            "notes": notes,
            "decision_time": decision_time,
        }
        append_review_log(review_path, record)
        st.success(f"Saved decision: {decision}")

    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("Approve", use_container_width=True):
            write_decision("Approve")
    with b2:
        if st.button("Reject", use_container_width=True):
            write_decision("Reject")
    with b3:
        if st.button("Needs Review", use_container_width=True):
            write_decision("Needs Review")


def page_metrics(cfg: Dict[str, Any]):
    st.subheader("Metrics")

    review_path = cfg_get(cfg, "paths", "review_log_parquet", default="data/review_log.parquet")
    df = load_review_log(review_path)
    if df.empty:
        st.info("No review decisions yet.")
        return

    st.markdown("### Decisions")
    counts = df["analyst_decision"].value_counts()
    st.bar_chart(counts)

    st.markdown("### Average ensemble score by decision")
    if "ensemble_score" in df.columns:
        avg = df.groupby("analyst_decision")["ensemble_score"].mean().sort_values(ascending=False)
        st.bar_chart(avg)

    st.markdown("### Time to decision (seconds)")
    # if you later store "created_time" you can compute true latency. For now, decision_time only.
    if "decision_time" in df.columns and "timestamp" in df.columns:
        dt1 = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        dt2 = pd.to_datetime(df["decision_time"], errors="coerce", utc=True)
        delta = (dt2 - dt1).dt.total_seconds()
        delta = delta.replace([float("inf"), float("-inf")], pd.NA).dropna()
        if not delta.empty:
            st.line_chart(delta.reset_index(drop=True))
        else:
            st.caption("Not enough timestamp info to compute time-to-decision.")


# ----------------------------
# Main
# ----------------------------

def main():
    st.set_page_config(page_title="FPN Review Dashboard", layout="wide")
    cfg = load_config()

    st.title(cfg_get(cfg, "app", "title", default="FPN Review Dashboard"))

    mode, analyst, min_score, max_rows = sidebar(cfg)

    page = st.session_state.get("page", "Queue")
    if page == "Queue":
        page_queue(cfg, mode=mode, min_score=min_score, max_rows=max_rows)
    elif page == "Details":
        page_details(cfg, mode=mode, analyst=analyst, min_score=min_score, max_rows=max_rows)
    else:
        page_metrics(cfg)


if __name__ == "__main__":
    main()
