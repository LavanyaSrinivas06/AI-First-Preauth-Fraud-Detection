import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from api.core.errors import ApiError
from api.services import model_service


def test_sha256_dict_stable_hash():
    d1 = {"b": 2, "a": 1}
    d2 = {"a": 1, "b": 2}
    assert model_service._sha256_dict(d1) == model_service._sha256_dict(d2)


@pytest.mark.parametrize("raw,expected", [
    ("", "v1"),
    ("   ", "v1"),
    ("unknown", "v1"),
    ("n/a", "v1"),
    ("legacy-unknown", "v1"),
    ("2026-01-01T10:10:10Z", "v1"),  # ISO timestamp gets normalized
    ("xgb-feedback-2026w01", "xgb-feedback-2026w01"),
    ("xgb feedback 2026w01", "xgb-feedback-2026w01"),  # unsafe chars become '-'
])
def test_normalize_model_version(raw, expected):
    assert model_service._normalize_model_version(raw) == expected


def test_load_active_xgb_registry_missing_file(tmp_path):
    art_dir = tmp_path
    with pytest.raises(ApiError) as e:
        model_service._load_active_xgb_registry(art_dir)
    assert e.value.code == "model_registry_missing"


def test_load_active_xgb_registry_invalid_json(tmp_path):
    p = tmp_path / "models"
    p.mkdir()
    f = p / "active_xgb.json"
    f.write_text("{not-json", encoding="utf-8")

    with pytest.raises(ApiError) as e:
        model_service._load_active_xgb_registry(tmp_path)
    assert e.value.code == "model_registry_invalid"


def test_load_active_xgb_registry_missing_active_model_key(tmp_path):
    p = tmp_path / "models"
    p.mkdir()
    f = p / "active_xgb.json"
    f.write_text(json.dumps({"version": "xgb-feedback-2026w01"}), encoding="utf-8")

    with pytest.raises(ApiError) as e:
        model_service._load_active_xgb_registry(tmp_path)
    assert e.value.code == "model_registry_invalid"


def test_load_active_xgb_registry_rejects_non_xgb_version(tmp_path):
    p = tmp_path / "models"
    p.mkdir()
    f = p / "active_xgb.json"
    f.write_text(json.dumps({"active_model": "xgb_model.pkl", "version": "my-model-1"}), encoding="utf-8")

    with pytest.raises(ApiError) as e:
        model_service._load_active_xgb_registry(tmp_path)
    assert e.value.code == "model_registry_error"


def test_load_model_schema_from_processed_train_happy(tmp_path, monkeypatch):
    # Fake Settings with data_dir pointing to tmp_path
    class S:
        data_dir = str(tmp_path)

    train = tmp_path / "train.csv"
    cols = [f"F{i}" for i in range(102)]
    df = pd.DataFrame([[0] * 103], columns=cols + ["Class"])
    df.to_csv(train, index=False)

    feats = model_service._load_model_schema_from_processed_train(S())
    assert len(feats) == 102
    assert "Class" not in feats


def test_load_model_schema_from_processed_train_wrong_feature_count(tmp_path):
    class S:
        data_dir = str(tmp_path)

    train = tmp_path / "train.csv"
    cols = [f"F{i}" for i in range(10)]
    df = pd.DataFrame([[0] * 11], columns=cols + ["Class"])
    df.to_csv(train, index=False)

    with pytest.raises(ApiError) as e:
        model_service._load_model_schema_from_processed_train(S())
    assert e.value.code == "artifact_invalid"


def test_ae_percentile_and_bucket():
    # minimal fake art
    class A:
        ae_block = 3.0
        ae_review = 1.0
        ae_legit_sorted_errors = np.array([0.1, 0.5, 1.0, 2.0, 10.0], dtype="float32")

    art = A()

    assert model_service.ae_bucket(art, 0.2) == "normal"
    assert model_service.ae_bucket(art, 1.0) == "elevated"
    assert model_service.ae_bucket(art, 5.0) == "extreme"

    pct = model_service.ae_percentile_vs_legit(art, 2.0)
    assert 0.0 <= pct <= 100.0
