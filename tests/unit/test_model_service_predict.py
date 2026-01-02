import numpy as np
import pandas as pd
import pytest

from api.services import model_service
from api.core.errors import ApiError


def test_predict_from_processed_102_returns_no_ae_outside_grayzone(monkeypatch):
    class DummySettings:
        xgb_t_low = 0.05
        xgb_t_high = 0.8

    class DummyArt:
        model_features = ["a", "b"]

    # ensure_loaded returns DummyArt
    monkeypatch.setattr(model_service, "ensure_loaded", lambda settings: DummyArt())
    # XGB score outside gray-zone
    monkeypatch.setattr(model_service, "score_xgb", lambda art, X_df: 0.95)

    out = model_service.predict_from_processed_102(DummySettings(), {"a": 1, "b": 2})
    p_xgb, ae_err, payload_hash, ae_pct, ae_bkt = out

    assert p_xgb == 0.95
    assert ae_err is None
    assert ae_pct is None
    assert ae_bkt == "n/a"
    assert isinstance(payload_hash, str) and len(payload_hash) > 10


def test_predict_from_processed_102_runs_ae_in_grayzone(monkeypatch):
    class DummySettings:
        xgb_t_low = 0.05
        xgb_t_high = 0.8

    class DummyArt:
        model_features = ["a", "b"]
        ae_review = 0.1
        ae_block = 0.3
        ae_legit_sorted_errors = np.array([0.01, 0.02, 0.03], dtype="float32")

    monkeypatch.setattr(model_service, "ensure_loaded", lambda settings: DummyArt())
    monkeypatch.setattr(model_service, "score_xgb", lambda art, X_df: 0.5)  # gray-zone
    monkeypatch.setattr(model_service, "ae_reconstruction_error", lambda art, X_dense: 0.2)

    out = model_service.predict_from_processed_102(DummySettings(), {"a": 1, "b": 2})
    p_xgb, ae_err, payload_hash, ae_pct, ae_bkt = out

    assert p_xgb == 0.5
    assert ae_err == 0.2
    assert ae_bkt in {"normal", "elevated", "extreme"}
    assert ae_pct is not None


def test_predict_from_processed_102_missing_required_feature_raises(monkeypatch):
    class DummySettings:
        xgb_t_low = 0.05
        xgb_t_high = 0.8

    class DummyArt:
        model_features = ["a", "b"]

    monkeypatch.setattr(model_service, "ensure_loaded", lambda settings: DummyArt())

    with pytest.raises(ApiError) as e:
        model_service.predict_from_processed_102(DummySettings(), {"a": 1})
    assert e.value.code == "missing_required_field"
