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


def sidebar(cfg: Dict[str, Any]) -> Tuple[str, str]:
    st.sidebar.markdown("## Navigation")

    page = st.sidebar.radio("Page", ["Ops", "Queue", "Log"], index=0)

    st.sidebar.divider()
    st.sidebar.markdown("## Settings")

    api_base = str(cfg_get(cfg, "api", "base_url", default="http://127.0.0.1:8000")).rstrip("/")
    api_base = st.sidebar.text_input("API base URL", value=api_base)

    st.sidebar.caption("Tip: Keep API running: `uvicorn api.main:app --reload`")

    return page, api_base


def main():
    st.set_page_config(page_title="FPN Review Dashboard", layout="wide")
    cfg = load_config()

    st.title(cfg_get(cfg, "app", "title", default="FPN Review Dashboard"))

    page, api_base = sidebar(cfg)

    if page == "Ops":
        from dashboard.screens.ops import render_ops_page
        render_ops_page(cfg, api_base=api_base)
    elif page == "Queue":
        from dashboard.screens.queue import render_queue_page
        render_queue_page(cfg, api_base=api_base)
    else:
        from dashboard.screens.log import render_log_page
        render_log_page(cfg, api_base=api_base)


if __name__ == "__main__":
    main()
