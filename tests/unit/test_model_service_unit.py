# tests/unit/test_model_service_unit.py
import numpy as np
import pytest

from api.core.errors import ApiError
from api.services import model_service


def test_sha256_dict_is_stable():
    d1 = {"b": 2, "a": 1}
    d2 = {"a": 1, "b": 2}
    assert model_service._sha256_dict(d1) == model_service._sha256_dict(d2)


def test_ensure_numpy_dense_from_numpy():
    X = np.array([[1.0, 2.0]], dtype=np.float64)
    out = model_service._ensure_numpy_dense(X)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float32
    assert out.shape == (1, 2)


def test_validate_processed_payload_missing_raises_apierror():
    payload = {"a": 1}
    with pytest.raises(ApiError) as e:
        model_service._validate_processed_payload(payload, ["a", "b"])
    assert e.value.status_code == 400


def test_validate_processed_payload_extra_key_allowed():
    # Current implementation only checks for missing keys.
    payload = {"num__V1": 0.0, "cat__x": 1, "extra": 123}
    model_features = ["num__V1", "cat__x"]
    model_service._validate_processed_payload(payload, model_features)  # should NOT raise
