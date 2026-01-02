# tests/unit/test_model_service_full.py
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from api.core.config import Settings
from api.core.errors import ApiError
from api.services import model_service


class DummyXGB:
    def __init__(self, p: float):
        self.p = float(p)

    def predict_proba(self, X):
        # shape: (n, 2)
        n = len(X)
        return np.tile(np.array([[1.0 - self.p, self.p]], dtype="float32"), (n, 1))


class DummyAE:
    def __init__(self):
        self.calls = 0

    def predict(self, X, verbose=0):
        self.calls += 1
        # return something deterministic with same shape
        return np.zeros_like(X, dtype="float32")


def _write_processed_train_csv(data_dir: Path, feature_names: list[str]) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    cols = feature_names + ["Class"]
    df = pd.DataFrame([[0.0] * len(feature_names) + [0]], columns=cols)
    (data_dir / "train.csv").write_text(df.to_csv(index=False), encoding="utf-8")


def _write_val_csv_all_legit(data_dir: Path, feature_names: list[str], nrows: int = 5) -> None:
    cols = feature_names + ["Class"]
    rows = []
    for _ in range(nrows):
        rows.append([0.1] * len(feature_names) + [0])
    df = pd.DataFrame(rows, columns=cols)
    (data_dir / "val.csv").write_text(df.to_csv(index=False), encoding="utf-8")


def _mk_settings(tmp_path: Path) -> Settings:
    # Settings in your project already supports repo_root and uses artifacts_path()/data_dir
    return Settings(repo_root=str(tmp_path))


def _mk_artifacts_layout(tmp_path: Path, settings: Settings) -> tuple[Path, Path, Path]:
    art_dir = settings.artifacts_path()
    models_dir = art_dir / "models"
    thresholds_path = art_dir / settings.ae_thresholds_path
    ae_model_path = art_dir / settings.ae_model_path

    models_dir.mkdir(parents=True, exist_ok=True)
    thresholds_path.parent.mkdir(parents=True, exist_ok=True)
    ae_model_path.parent.mkdir(parents=True, exist_ok=True)

    return art_dir, thresholds_path, ae_model_path


@pytest.fixture(autouse=True)
def _reset_artifacts_cache():
    # ensure no cross-test cache pollution
    model_service._ART = None
    yield
    model_service._ART = None


def test_normalize_model_version_variants():
    n = model_service._normalize_model_version
    assert n("") == "v1"
    assert n("   ") == "v1"
    assert n("unknown") == "v1"
    assert n("legacy-unknown") == "v1"
    assert n("2025-01-01T10:10:10Z") == "v1"
    assert n("xgb-feedback-2026w01") == "xgb-feedback-2026w01"
    assert n("weird value !!!") == "weird-value"


def test_load_active_xgb_registry_missing_pointer(tmp_path: Path):
    settings = _mk_settings(tmp_path)
    art_dir = settings.artifacts_path()
    with pytest.raises(ApiError) as e:
        model_service._load_active_xgb_registry(art_dir)
    assert e.value.code == "model_registry_missing"


def test_load_active_xgb_registry_invalid_json(tmp_path: Path):
    settings = _mk_settings(tmp_path)
    art_dir = settings.artifacts_path()
    p = art_dir / "models" / "active_xgb.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ApiError) as e:
        model_service._load_active_xgb_registry(art_dir)
    assert e.value.code == "model_registry_invalid"


def test_load_active_xgb_registry_invalid_version_rejected(tmp_path: Path):
    settings = _mk_settings(tmp_path)
    art_dir = settings.artifacts_path()
    p = art_dir / "models" / "active_xgb.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"active_model": "xgb_model.pkl", "version": "badtag-123"}), encoding="utf-8")
    with pytest.raises(ApiError) as e:
        model_service._load_active_xgb_registry(art_dir)
    assert e.value.code == "model_registry_error"


def test_load_ae_thresholds_invalid_json(tmp_path: Path):
    settings = _mk_settings(tmp_path)
    art_dir, thresholds_path, _ = _mk_artifacts_layout(tmp_path, settings)
    thresholds_path.write_text("{bad", encoding="utf-8")
    with pytest.raises(ApiError) as e:
        model_service._load_ae_thresholds(art_dir, settings)
    assert e.value.code == "artifact_invalid"


def test_load_model_schema_wrong_feature_count(tmp_path: Path):
    settings = _mk_settings(tmp_path)
    data_dir = Path(settings.data_dir)

    # wrong count (2) instead of 102
    df = pd.DataFrame([[0.1, 0.2, 0]], columns=["f1", "f2", "Class"])
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.csv").write_text(df.to_csv(index=False), encoding="utf-8")

    with pytest.raises(ApiError) as e:
        model_service._load_model_schema_from_processed_train(settings)
    assert e.value.code == "artifact_invalid"


def test_ensure_loaded_happy_path_builds_and_caches(tmp_path: Path, monkeypatch):
    settings = _mk_settings(tmp_path)
    art_dir, thresholds_path, ae_model_path = _mk_artifacts_layout(tmp_path, settings)

    # model registry pointer
    (art_dir / "models" / "active_xgb.json").write_text(
        json.dumps({"active_model": "xgb_model.pkl", "version": "xgb-feedback-2026w01"}),
        encoding="utf-8",
    )

    # dummy model file must exist (joblib.load is patched, but ensure_loaded checks existence)
    (art_dir / "models" / "xgb_model.pkl").write_bytes(b"dummy")

    # thresholds
    thresholds_path.write_text(json.dumps({"review": 0.1, "block": 0.3}), encoding="utf-8")

    # dummy AE model path must exist (load_model patched, but file presence is still used by Keras sometimes)
    ae_model_path.write_bytes(b"dummy")

    # train/val schema (exactly 102)
    feats = [f"f{i}" for i in range(102)]
    _write_processed_train_csv(Path(settings.data_dir), feats)
    _write_val_csv_all_legit(Path(settings.data_dir), feats, nrows=3)

    dxgb = DummyXGB(p=0.5)
    dae = DummyAE()

    monkeypatch.setattr(model_service.joblib, "load", lambda p: dxgb)
    monkeypatch.setattr(model_service, "load_model", lambda p, compile=False: dae)

    art1 = model_service.ensure_loaded(settings)
    assert art1.model_version == "xgb-feedback-2026w01"
    assert len(art1.model_features) == 102
    assert art1.ae_review == 0.1
    assert art1.ae_block == 0.3
    assert isinstance(art1.ae_legit_sorted_errors, np.ndarray)
    assert len(art1.ae_legit_sorted_errors) > 0

    # cached
    art2 = model_service.ensure_loaded(settings)
    assert art2 is art1


def test_predict_from_processed_102_outside_grayzone_skips_ae(tmp_path: Path, monkeypatch):
    settings = _mk_settings(tmp_path)
    art_dir, thresholds_path, ae_model_path = _mk_artifacts_layout(tmp_path, settings)

    (art_dir / "models" / "active_xgb.json").write_text(
        json.dumps({"active_model": "xgb_model.pkl", "version": "xgb-feedback-2026w01"}),
        encoding="utf-8",
    )
    (art_dir / "models" / "xgb_model.pkl").write_bytes(b"dummy")
    thresholds_path.write_text(json.dumps({"review": 0.1, "block": 0.3}), encoding="utf-8")
    ae_model_path.write_bytes(b"dummy")

    feats = [f"f{i}" for i in range(102)]
    _write_processed_train_csv(Path(settings.data_dir), feats)
    _write_val_csv_all_legit(Path(settings.data_dir), feats, nrows=2)

    # outside grayzone: p_xgb >= xgb_t_high (default xgb_t_high=0.8 in your Settings)
    dxgb = DummyXGB(p=0.95)
    dae = DummyAE()
    monkeypatch.setattr(model_service.joblib, "load", lambda p: dxgb)
    monkeypatch.setattr(model_service, "load_model", lambda p, compile=False: dae)

    payload = {k: 0.0 for k in feats}
    p_xgb, ae_err, payload_hash, ae_pct, ae_bkt = model_service.predict_from_processed_102(settings, payload)

    assert p_xgb >= settings.xgb_t_high
    assert ae_err is None
    assert ae_pct is None
    assert ae_bkt == "n/a"
    assert isinstance(payload_hash, str) and len(payload_hash) == 64
    assert dae.calls == 0


def test_predict_from_processed_102_in_grayzone_runs_ae(tmp_path: Path, monkeypatch):
    settings = _mk_settings(tmp_path)
    art_dir, thresholds_path, ae_model_path = _mk_artifacts_layout(tmp_path, settings)

    (art_dir / "models" / "active_xgb.json").write_text(
        json.dumps({"active_model": "xgb_model.pkl", "version": "xgb-feedback-2026w01"}),
        encoding="utf-8",
    )
    (art_dir / "models" / "xgb_model.pkl").write_bytes(b"dummy")
    thresholds_path.write_text(json.dumps({"review": 0.1, "block": 0.3}), encoding="utf-8")
    ae_model_path.write_bytes(b"dummy")

    feats = [f"f{i}" for i in range(102)]
    _write_processed_train_csv(Path(settings.data_dir), feats)
    _write_val_csv_all_legit(Path(settings.data_dir), feats, nrows=5)

    dxgb = DummyXGB(p=0.5)  # inside [low, high)
    dae = DummyAE()
    monkeypatch.setattr(model_service.joblib, "load", lambda p: dxgb)
    monkeypatch.setattr(model_service, "load_model", lambda p, compile=False: dae)

    payload = {k: 0.2 for k in feats}
    p_xgb, ae_err, payload_hash, ae_pct, ae_bkt = model_service.predict_from_processed_102(settings, payload)

    assert settings.xgb_t_low <= p_xgb < settings.xgb_t_high
    assert ae_err is not None
    assert ae_pct is not None
    assert ae_bkt in {"normal", "elevated", "extreme"}
    assert dae.calls >= 1


def test_load_or_build_ae_baseline_uses_existing_npy(tmp_path: Path, monkeypatch):
    settings = _mk_settings(tmp_path)
    art_dir, thresholds_path, ae_model_path = _mk_artifacts_layout(tmp_path, settings)

    baseline_path = art_dir / settings.ae_baseline_path
    baseline_path.parent.mkdir(parents=True, exist_ok=True)

    # write an unsorted baseline to verify it gets sorted
    np.save(baseline_path, np.array([0.3, 0.1, 0.2], dtype="float32"))

    dae = DummyAE()
    base = model_service._load_or_build_ae_baseline(art_dir, settings, dae)
    assert list(base) == sorted(list(base))
