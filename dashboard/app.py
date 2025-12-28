from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import streamlit as st
import yaml

# --- Ensure repo root is on PYTHONPATH so "import dashboard.*" works ---
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CONFIG_PATH = Path("dashboard/config.yaml")


@st.cache_data(show_spinner=False)
def load_config() -> Dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {}
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}


def cfg_get(cfg: Dict[str, Any], *keys: str, default=None):
    cur: Any = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def sidebar(cfg: Dict[str, Any]) -> Tuple[str, float, int, str]:
    st.sidebar.header("Settings")

    api_base = str(cfg_get(cfg, "api", "base_url", default="http://127.0.0.1:8000")).rstrip("/")
    st.sidebar.text_input("API base URL", value=api_base, key="api_base")

    st.sidebar.divider()
    st.sidebar.subheader("Queue filters")

    min_default = float(cfg_get(cfg, "queue", "min_score_xgb", default=0.0))
    min_score = st.sidebar.slider("Min XGB prob (for viewing)", 0.0, 1.0, min_default, 0.01)

    max_rows = int(cfg_get(cfg, "queue", "max_rows", default=200))

    st.sidebar.divider()
    page = st.sidebar.radio("Pages", ["Queue", "Log"], index=0)

    return page, float(min_score), int(max_rows), st.session_state["api_base"]


def main():
    st.set_page_config(page_title="FPN Review Dashboard", layout="wide")
    cfg = load_config()

    st.title(cfg_get(cfg, "app", "title", default="FPN Review Dashboard"))

    page, min_score, max_rows, api_base = sidebar(cfg)

    if page == "Queue":
        from dashboard.pages.queue import render_queue_page
        render_queue_page(cfg, api_base=api_base, min_score=min_score, max_rows=max_rows)
    else:
        from dashboard.pages.log import render_log_page
        render_log_page(cfg, api_base=api_base)


if __name__ == "__main__":
    main()
