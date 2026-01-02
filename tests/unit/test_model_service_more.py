# tests/unit/test_model_service_more.py
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from api.core.errors import ApiError
from api.services import model_service


def test_normalize_model_version_basics():
    n = model_service._normalize_model_version
    assert n("") == "v1"
    assert n("   ") == "v1"
    assert n("unknown") == "v1"
    assert n("n/a") == "v1"
    assert n("legacy-unknown") == "v1"
    assert n("2026-01-01T10:20:30Z") == "v1"
    assert n("xgb-feedback-2026w01") == "xgb-feedback-2026w01"
    assert n("xgb feedback 2026w01") == "xgb-feedback-2026w01"


def test_ensure_numpy_dense_from_sparse_like():
    class SparseLike:
        def __init__(self, arr):
            self._arr = arr
        def toarray(self):
            return self._arr

    X = SparseLike(np.array([[1, 2]], dtype=np.float64))
    out = model_service._ensure_numpy_dense(X)
    assert out.dtype == np.float32
    assert out.shape == (1, 2)


def test_ae_percentile_vs_legit_none_when_empty():
    art = type("A", (), {"ae_legit_sorted_errors": np.array([], dtype="float32")})()
    assert model_service.ae_percentile_vs_legit(art, 1.0) is None


def test_ae_percentile_vs_legit_rank():
    base = np.array([0.1, 0.2, 0.3, 0.4], dtype="float32")
    art = type("A", (), {"ae_legit_sorted_errors": base})()
    # ae_err=0.2 should rank "right" -> index 2 -> 50%
    pct = model_service.ae_percentile_vs_legit(art, 0.2)
    assert pct == 50.0


def test_ae_bucket_thresholds():
    art = type("A", (), {"ae_review": 0.5, "ae_block": 1.5})()
    assert model_service.ae_bucket(art, 0.1) == "normal"
    assert model_service.ae_bucket(art, 0.5) == "elevated"
    assert model_service.ae_bucket(art, 2.0) == "extreme"


def test_load_active_xgb_registry_missing_raises(tmp_path):
    with pytest.raises(ApiError) as e:
        model_service._load_active_xgb_registry(tmp_path)
    assert e.value.code == "model_registry_missing"


def test_load_active_xgb_registry_invalid_json_raises(tmp_path):
    p = tmp_path / "models" / "active_xgb.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{not-json", encoding="utf-8")
    with pytest.raises(ApiError) as e:
        model_service._load_active_xgb_registry(tmp_path)
    assert e.value.code == "model_registry_invalid"


def test_load_active_xgb_registry_invalid_version_rejected(tmp_path):
    p = tmp_path / "models" / "active_xgb.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"active_model": "xgb_model.pkl", "version": "weird!!"}), encoding="utf-8")
    with pytest.raises(ApiError) as e:
        model_service._load_active_xgb_registry(tmp_path)
    assert e.value.code == "model_registry_error"


def test_load_active_xgb_registry_valid(tmp_path):
    p = tmp_path / "models" / "active_xgb.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"active_model": "xgb_model.pkl", "version": "xgb-feedback-2026w01"}), encoding="utf-8")
    obj = model_service._load_active_xgb_registry(tmp_path)
    assert obj["active_model"] == "xgb_model.pkl"
    assert obj["version"] == "xgb-feedback-2026w01"


def test_predict_from_processed_102_short_circuits_without_ae(monkeypatch, tmp_path):
    """
    Cover branch: if p_xgb outside [t_low, t_high) -> returns without AE.
    We monkeypatch ensure_loaded + score_xgb to avoid real models/files.
    """
    from api.core.config import Settings

    # Dummy artifacts
    class DummyArt:
        model_features = ["f1", "f2"]
        ae_review = 0.1
        ae_block = 0.2
        ae_legit_sorted_errors = np.array([0.1, 0.2], dtype="float32")
        model_version = "xgb-feedback-2026w01"
        xgb = object()
        ae = object()

    monkeypatch.setattr(model_service, "ensure_loaded", lambda settings: DummyArt())
    monkeypatch.setattr(model_service, "score_xgb", lambda art, X_df: 0.99)

    s = Settings(repo_root=str(tmp_path))
    payload = {"f1": 1.0, "f2": 2.0}

    p_xgb, ae_err, payload_hash, ae_pct, ae_bkt = model_service.predict_from_processed_102(s, payload)
    assert p_xgb == 0.99
    assert ae_err is None
    assert ae_pct is None
    assert ae_bkt == "n/a"
    assert isinstance(payload_hash, str) and len(payload_hash) == 64


def test_predict_from_processed_102_runs_ae_path(monkeypatch, tmp_path):
    """
    Cover branch: p_xgb in gray-zone -> runs AE helpers.
    """
    from api.core.config import Settings

    class DummyArt:
        model_features = ["f1", "f2"]
        ae_review = 0.5
        ae_block = 1.5
        ae_legit_sorted_errors = np.array([0.1, 0.2, 0.3, 0.4], dtype="float32")
        model_version = "xgb-feedback-2026w01"
        xgb = object()
        ae = object()

    monkeypatch.setattr(model_service, "ensure_loaded", lambda settings: DummyArt())
    monkeypatch.setattr(model_service, "score_xgb", lambda art, X_df: 0.5)
    monkeypatch.setattr(model_service, "ae_reconstruction_error", lambda art, X_dense: 0.2)

    s = Settings(repo_root=str(tmp_path))
    payload = {"f1": 1.0, "f2": 2.0}

    p_xgb, ae_err, payload_hash, ae_pct, ae_bkt = model_service.predict_from_processed_102(s, payload)
    assert p_xgb == 0.5
    assert ae_err == 0.2
    assert ae_pct == 50.0  # from base + searchsorted(right) in ae_percentile_vs_legit
    assert ae_bkt == "normal"
    assert isinstance(payload_hash, str) and len(payload_hash) == 64
