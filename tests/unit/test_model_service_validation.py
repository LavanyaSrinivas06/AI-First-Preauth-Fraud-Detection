# tests/unit/test_model_service_validation.py
import pytest

from api.core.errors import ApiError
from api.services import model_service


def test_validate_processed_payload_accepts_exact_features():
    payload = {"a": 1, "b": 2}
    model_service._validate_processed_payload(payload, ["a", "b"])  # should not raise


def test_validate_processed_payload_rejects_missing_feature():
    payload = {"a": 1}
    with pytest.raises(ApiError):
        model_service._validate_processed_payload(payload, ["a", "b"])


def test_validate_processed_payload_allows_extra_features():
    # Current behavior: extra keys allowed
    payload = {"a": 1, "b": 2, "c": 3}
    model_service._validate_processed_payload(payload, ["a", "b"])  # should not raise
