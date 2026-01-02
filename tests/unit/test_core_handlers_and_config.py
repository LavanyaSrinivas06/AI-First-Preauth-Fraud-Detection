# tests/unit/test_core_handlers_and_config.py
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.core.config import Settings
from api.core.errors import ApiError, register_exception_handlers


def test_settings_abs_paths_methods(tmp_path):
    s = Settings(repo_root=str(tmp_path))

    # These exist in your function coverage list (0% before)
    assert str(s.abs_feedback_log_path()).endswith(str(s.feedback_log_path))
    assert str(s.abs_review_queue_jsonl_path()).endswith(str(s.review_queue_jsonl_path))
    assert str(s.abs_xgb_model_path()).endswith(str(s.xgb_model_path))
    assert str(s.abs_xgb_model_feedback_path()).endswith(str(s.xgb_model_feedback_path))
    assert str(s.abs_ae_model_path()).endswith(str(s.ae_model_path))
    assert str(s.abs_preprocess_path()).endswith(str(s.preprocess_path))
    assert str(s.abs_features_path()).endswith(str(s.features_path))
    assert str(s.abs_ae_thresholds_path()).endswith(str(s.ae_thresholds_path))
    assert str(s.abs_ae_baseline_path()).endswith(str(s.ae_baseline_path))


def test_register_exception_handlers_apierror_shape():
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/boom")
    def boom():
        raise ApiError(400, "bad_request", "nope", param="x")

    client = TestClient(app)
    r = client.get("/boom")
    assert r.status_code == 400
    j = r.json()
    assert "error" in j
    assert j["error"]["code"] == "bad_request"
    assert j["error"]["message"] == "nope"
    assert j["error"]["param"] == "x"


def test_register_exception_handlers_unhandled_shape():
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/crash")
    def crash():
        raise RuntimeError("kaboom")

    client = TestClient(app)
    r = client.get("/crash")
    assert r.status_code == 500
    j = r.json()
    assert j["error"]["code"] == "internal_error"
    assert "RuntimeError" in j["error"]["message"]
