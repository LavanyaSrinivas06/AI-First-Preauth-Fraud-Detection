# tests/unit/test_errors.py
from api.core.errors import ApiError, stripe_error


def test_api_error_fields():
    e = ApiError(400, "bad_request", "nope", param="x")
    assert e.status_code == 400
    assert e.code == "bad_request"
    assert e.message == "nope"
    assert e.param == "x"


def test_stripe_error_shape_with_param():
    body = stripe_error("bad_request", "nope", param="x")
    assert "error" in body
    assert body["error"]["code"] == "bad_request"
    assert body["error"]["message"] == "nope"
    assert body["error"]["param"] == "x"


def test_stripe_error_shape_without_param():
    body = stripe_error("bad_request", "nope")
    assert "error" in body
    assert body["error"]["code"] == "bad_request"
    assert body["error"]["message"] == "nope"
    assert "param" not in body["error"]
