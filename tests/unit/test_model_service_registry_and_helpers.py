import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from api.core.errors import ApiError
from api.services import model_service


def test_normalize_model_version_basics():
    f = model_service._normalize_model_version
    assert f("") == "v1"
    assert f("   ") == "v1"
    assert f("unknown") == "v1"
    assert f("n/a") == "v1"
    assert f("legacy-unknown") == "v1"
    assert f("2026-01-02T10:11:12Z") == "v1"  # ISO -> v1
    assert f("xgb-feedback-2026w01") == "xgb-feedback-2026w01"
    assert f("xgb feedback 2026 w01") == "xgb-feedback-2026-w01"


def test_sha256_dict_stable():
    d1 = {"b": 2, "a": 1}
    d2 = {"a": 1, "b": 2}
    assert model_service._sha256_dict(d1) == model_service._sha256_dict(d2)


def test_ensure_numpy_dense_accepts_ndarray():
    x = np.array([[1, 2]], dtype=np.float64)
    out = model_service._ensure_numpy_dense(x)
    assert out.dtype == np.float32
    assert out.shape == (1, 2)


def test_load_ae_thresholds_reads_json(tmp_path):
    class S:
        ae_thresholds_path = "thresholds/ae_thresholds.json"

    art_dir = tmp_path
    p = art_dir / "thresholds" / "ae_thresholds.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"review": 0.1, "block": 0.3}), encoding="utf-8")

    review, block = model_service._load_ae_thresholds(art_dir, S())
    assert review == 0.1
    assert block == 0.3


def test_load_model_schema_from_processed_train_reads_102(tmp_path):
    # build a fake train.csv with 102 feature columns + Class
    class S:
        data_dir = str(tmp_path)

    cols = [f"f{i}" for i in range(102)]
    df = pd.DataFrame([{**{c: 0 for c in cols}, "Class": 0}])
    (tmp_path / "train.csv").write_text(df.to_csv(index=False), encoding="utf-8")

    feats = model_service._load_model_schema_from_processed_train(S())
    assert len(feats) == 102
    assert "Class" not in feats


def test_load_active_xgb_registry_ok(tmp_path):
    art_dir = tmp_path
    p = art_dir / "models" / "active_xgb.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"active_model": "xgb_model.pkl", "version": "xgb-feedback-2026w01"}), encoding="utf-8")

    obj = model_service._load_active_xgb_registry(art_dir)
    assert obj["active_model"] == "xgb_model.pkl"
    assert obj["version"] == "xgb-feedback-2026w01"


def test_load_active_xgb_registry_rejects_non_xgb_version(tmp_path):
    art_dir = tmp_path
    p = art_dir / "models" / "active_xgb.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"active_model": "xgb_model.pkl", "version": "weird-version"}), encoding="utf-8")

    with pytest.raises(ApiError) as e:
        model_service._load_active_xgb_registry(art_dir)
    assert e.value.status_code == 500
    assert e.value.code in {"model_registry_error", "model_registry_invalid"}
