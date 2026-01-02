# tests/conftest.py
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# now imports work
from api.main import app  # noqa


import json
from typing import Dict, Any

import pytest
from fastapi.testclient import TestClient

from api.core.config import get_settings, Settings
from api.services import store as store_mod


@pytest.fixture()
def tmp_repo(tmp_path: Path) -> Path:
    """
    Create a temp repo-like structure so tests don't touch your real artifacts/db.
    """
    (tmp_path / "artifacts" / "models").mkdir(parents=True, exist_ok=True)
    (tmp_path / "artifacts" / "thresholds").mkdir(parents=True, exist_ok=True)
    (tmp_path / "artifacts" / "ae_errors").mkdir(parents=True, exist_ok=True)
    (tmp_path / "artifacts" / "preprocess").mkdir(parents=True, exist_ok=True)
    (tmp_path / "artifacts" / "snapshots" / "feature_snapshots").mkdir(parents=True, exist_ok=True)
    (tmp_path / "artifacts" / "stores").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    return tmp_path


@pytest.fixture()
def test_settings(tmp_repo: Path) -> Settings:
    """
    Override Settings so API + services use temp paths.
    """
    s = Settings()
    s.repo_root = str(tmp_repo)
    s.artifacts_dir = "artifacts"
    s.data_dir = "data/processed"
    s.logs_dir = "logs"

    # stores
    s.sqlite_path = "artifacts/stores/inference_store.sqlite"
    s.feedback_log_path = "artifacts/stores/feedback_log.jsonl"
    s.review_queue_jsonl_path = "artifacts/stores/review_queue.jsonl"

    # models
    s.xgb_model_path = "models/xgb_model.pkl"
    s.xgb_model_feedback_path = "models/xgb_model_feedback.pkl"

    return s


@pytest.fixture()
def client(test_settings: Settings) -> TestClient:
    """
    FastAPI client with dependency override.
    """
    app.dependency_overrides[get_settings] = lambda: test_settings
    return TestClient(app)


@pytest.fixture()
def sqlite_path(test_settings: Settings) -> Path:
    return test_settings.abs_sqlite_path()


@pytest.fixture()
def init_test_db(sqlite_path: Path) -> Path:
    store_mod.init_db(sqlite_path)
    return sqlite_path


@pytest.fixture()
def sample_processed_102() -> Dict[str, Any]:
    """
    Minimal processed-102 payload substitute:
    In tests we don't need real ML outputs; we only need correct keys format.
    We'll load your existing fixture if present.
    """
    fp = Path("tests/fixtures/sample_processed_102.json")
    if fp.exists():
        return json.loads(fp.read_text(encoding="utf-8"))
    # fallback tiny placeholder (won't pass model_service validation unless your test stubs it)
    return {"num__V1": 0.0}
