import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from api.core.errors import ApiError
from api.services import model_service


# ------------------------
# helpers
# ------------------------
class DummyXGB:
    def __init__(self, p=0.9):
        self.p = p

    def predict_proba(self, X):
        # shape (n,2)
        return np.array([[1.0 - self.p, self.p]], dtype="float32")


class DummyAE:
    def __init__(self, rec_error=0.0):
        self.rec_error = rec_error

    def predict(self, X, verbose=0):
        # produce reconstruction with controlled error
        if self.rec_error == 0.0:
            return X.copy()
        return X + (self.rec_error ** 0.5)


class DummySettings:
    """
    Minimal Settings-like object with the attributes model_service.py needs.
    (avoids depending on your full Settings implementation)
    """
    def __init__(self, repo_root: Path, data_dir: Path, artifacts_dir: Path):
        self.repo_root = str(repo_root)
        self.data_dir = str(data_dir)

        # model_service uses these relative paths under artifacts_path()
        self.ae_thresholds_path = Path("thresholds/ae_thresholds.json")
        self.ae_model_path = Path("models/autoencoder_model.keras")
        self.ae_baseline_path = Path("ae_errors/ae_baseline_legit_errors.npy")

        # thresholds for "gray-zone"
        self.xgb_t_low = 0.05
        self.xgb_t_high = 0.8

        self._artifacts_dir = artifacts_dir

    def artifacts_path(self) -> Path:
        return self._artifacts_dir


@pytest.fixture()
def tmp_env(tmp_path: Path):
    repo = tmp_path
    data_dir = repo / "data" / "processed"
    art_dir = repo / "artifacts"
    (art_dir / "models").mkdir(parents=True)
    (art_dir / "thresholds").mkdir(parents=True)
    (art_dir / "ae_errors").mkdir(parents=True)
    data_dir.mkdir(parents=True)

    settings = DummySettings(repo, data_dir, art_dir)
    return settings, data_dir, art_dir


@pytest.fixture(autouse=True)
def reset_art_cache():
    model_service._ART = None
    yield
    model_service._ART = None


# ------------------------
# tests for small helpers
# ------------------------
def test_sha256_dict_is_stable():
    d1 = {"b": 2, "a": 1}
    d2 = {"a": 1, "b": 2}
    assert model_service._sha256_dict(d1) == model_service._sha256_dict(d2)


@pytest.mark.parametrize(
    "raw,expected_prefix",
    [
        ("", "v1"),
        ("unknown", "v1"),
        ("legacy-unknown", "v1"),
        ("n/a", "v1"),
        ("2026-01-02T12:34:56Z", "v1"),
        ("xgb-feedback-2026w01", "xgb-feedback-2026w01"),
        ("xgb feedback 2026w01", "xgb-feedback-2026w01"),
    ],
)
def test_normalize_model_version(raw, expected_prefix):
    out = model_service._normalize_model_version(raw)
    assert out.startswith(expected_prefix)


def test_validate_processed_payload_missing_raises():
    with pytest.raises(ApiError) as e:
        model_service._validate_processed_payload({"a": 1}, ["a", "b"])
    assert e.value.status_code == 400
    assert e.value.code == "missing_required_field"


def test_validate_processed_payload_extra_ok():
    # extra keys are allowed by design (only missing is blocked)
    model_service._validate_processed_payload({"a": 1, "b": 2, "c": 3}, ["a", "b"])


# ------------------------
# registry + thresholds + schema
# ------------------------
def test_load_active_xgb_registry_success(tmp_env):
    settings, data_dir, art_dir = tmp_env

    p = art_dir / "models" / "active_xgb.json"
    p.write_text(json.dumps({"active_model": "xgb_model.pkl", "version": "xgb-feedback-2026w01"}))

    out = model_service._load_active_xgb_registry(art_dir)
    assert out["active_model"] == "xgb_model.pkl"
    assert out["version"] == "xgb-feedback-2026w01"


def test_load_active_xgb_registry_missing_file_raises(tmp_env):
    _, _, art_dir = tmp_env
    with pytest.raises(ApiError) as e:
        model_service._load_active_xgb_registry(art_dir)
    assert e.value.code == "model_registry_missing"


def test_load_ae_thresholds_success(tmp_env):
    settings, _, art_dir = tmp_env
    p = art_dir / settings.ae_thresholds_path
    p.write_text(json.dumps({"review": 0.1, "block": 0.3}))
    r, b = model_service._load_ae_thresholds(art_dir, settings)
    assert r == 0.1
    assert b == 0.3


def test_load_model_schema_from_processed_train_success(tmp_env):
    settings, data_dir, _ = tmp_env
    # build minimal train.csv with 102 feature cols + Class
    cols = [f"f{i}" for i in range(102)] + ["Class"]
    df = pd.DataFrame([[0] * 103], columns=cols)
    df.to_csv(Path(settings.data_dir) / "train.csv", index=False)

    feats = model_service._load_model_schema_from_processed_train(settings)
    assert len(feats) == 102
    assert "Class" not in feats


# ------------------------
# baseline load/build
# ------------------------
def test_load_or_build_ae_baseline_loads_existing(tmp_env):
    settings, _, art_dir = tmp_env
    base_path = art_dir / settings.ae_baseline_path
    arr = np.array([0.3, 0.1, 0.2], dtype="float32")
    np.save(base_path, arr)

    base = model_service._load_or_build_ae_baseline(art_dir, settings, ae_model=DummyAE())
    assert np.allclose(base, np.array([0.1, 0.2, 0.3], dtype="float32"))


def test_ae_percentile_and_bucket():
    art = model_service.LoadedArtifacts(
        xgb=DummyXGB(0.9),
        ae=DummyAE(0.0),
        model_features=["a", "b"],
        ae_review=0.2,
        ae_block=0.5,
        ae_legit_sorted_errors=np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype="float32"),
        model_version="xgb-feedback-2026w01",
    )

    pct = model_service.ae_percentile_vs_legit(art, 0.2)
    assert 0.0 <= pct <= 100.0

    assert model_service.ae_bucket(art, 0.1) == "normal"
    assert model_service.ae_bucket(art, 0.2) == "elevated"
    assert model_service.ae_bucket(art, 0.6) == "extreme"


# ------------------------
# predict_from_processed_102 (both branches)
# ------------------------
def test_predict_from_processed_102_outside_grayzone_returns_na(tmp_env, monkeypatch):
    settings, data_dir, art_dir = tmp_env

    # minimal train schema
    cols = [f"f{i}" for i in range(102)] + ["Class"]
    pd.DataFrame([[0] * 103], columns=cols).to_csv(Path(settings.data_dir) / "train.csv", index=False)

    # registry + thresholds + baseline present (avoid heavy building)
    (art_dir / "models" / "active_xgb.json").write_text(
        json.dumps({"active_model": "xgb_model.pkl", "version": "xgb-feedback-2026w01"})
    )
    (art_dir / settings.ae_thresholds_path).write_text(json.dumps({"review": 0.1, "block": 0.3}))
    np.save(art_dir / settings.ae_baseline_path, np.array([0.0, 0.1, 0.2], dtype="float32"))

    # patch loaders
    monkeypatch.setattr(model_service.joblib, "load", lambda p: DummyXGB(p=0.99))  # HIGH => outside [0.05,0.8)
    monkeypatch.setattr(model_service, "load_model", lambda p, compile=False: DummyAE(0.0))
    # also create dummy file for existence check
    (art_dir / "models" / "xgb_model.pkl").write_text("dummy")
    (art_dir / settings.ae_model_path).write_text("dummy")

    payload = {f"f{i}": 0.0 for i in range(102)}
    p_xgb, ae_err, _, ae_pct, ae_bkt = model_service.predict_from_processed_102(settings, payload)

    assert p_xgb > 0.8
    assert ae_err is None
    assert ae_pct is None
    assert ae_bkt == "n/a"


def test_predict_from_processed_102_inside_grayzone_runs_ae(tmp_env, monkeypatch):
    settings, _, art_dir = tmp_env

    # schema
    cols = [f"f{i}" for i in range(102)] + ["Class"]
    pd.DataFrame([[0] * 103], columns=cols).to_csv(Path(settings.data_dir) / "train.csv", index=False)

    (art_dir / "models" / "active_xgb.json").write_text(
        json.dumps({"active_model": "xgb_model.pkl", "version": "xgb-feedback-2026w01"})
    )
    (art_dir / settings.ae_thresholds_path).write_text(json.dumps({"review": 0.1, "block": 0.3}))
    np.save(art_dir / settings.ae_baseline_path, np.array([0.0, 0.05, 0.1, 0.2, 0.4], dtype="float32"))

    monkeypatch.setattr(model_service.joblib, "load", lambda p: DummyXGB(p=0.5))  # inside gray zone
    monkeypatch.setattr(model_service, "load_model", lambda p, compile=False: DummyAE(0.0))
    (art_dir / "models" / "xgb_model.pkl").write_text("dummy")
    (art_dir / settings.ae_model_path).write_text("dummy")

    payload = {f"f{i}": 0.0 for i in range(102)}
    p_xgb, ae_err, _, ae_pct, ae_bkt = model_service.predict_from_processed_102(settings, payload)

    assert 0.05 <= p_xgb < 0.8
    assert ae_err == 0.0
    assert ae_pct is not None
    assert ae_bkt in {"normal", "elevated", "extreme"}
