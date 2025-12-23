import numpy as np
import pytest

import api.service as svc


class DummyXGB:
    def __init__(self, p: float):
        self.p = p

    def predict_proba(self, X):
        # returns [[p0, p1]]
        return np.array([[1.0 - self.p, self.p]], dtype=float)


class DummyPreprocess:
    def transform(self, df):
        # return already "processed" vector (1 x 3)
        return np.array([[0.1, 0.2, 0.3]], dtype=np.float32)


class DummyAE:
    def __init__(self, err: float):
        self.err = err

    def predict(self, X, verbose=0):
        # Create reconstruction that yields the desired MSE approx.
        # easiest: return X unchanged => err=0
        # For tests, we’ll just return X and monkeypatch _reconstruction_error instead.
        return X


def _minimal_features():
    return {
        "numerical_features": ["f1", "f2"],
        "categorical_features": ["c1"],
    }


def _minimal_payload():
    return {"f1": 1.0, "f2": 2.0, "c1": "A"}


def test_approve_when_low(monkeypatch):
    svc.XGB = DummyXGB(p=0.05)
    svc.PREPROCESS = DummyPreprocess()
    svc.AE = DummyAE(err=0.0)
    svc.FEATURES = _minimal_features()
    svc.AE_THRESHOLDS = {"review": 1.0, "block": 2.0}

    monkeypatch.setattr(svc, "_reconstruction_error", lambda *args, **kwargs: 0.1)

    out = svc.predict_hybrid(_minimal_payload())
    assert out["label"] in {"approve", "review", "block"}
    assert out["label"] == "approve"


def test_block_when_high(monkeypatch):
    svc.XGB = DummyXGB(p=0.95)
    svc.PREPROCESS = DummyPreprocess()
    svc.AE = DummyAE(err=0.0)
    svc.FEATURES = _minimal_features()
    svc.AE_THRESHOLDS = {"review": 1.0, "block": 2.0}

    monkeypatch.setattr(svc, "_reconstruction_error", lambda *args, **kwargs: 100.0)

    out = svc.predict_hybrid(_minimal_payload())
    assert out["label"] == "block"


def test_review_in_grayzone_with_ae(monkeypatch):
    # mid XGB, AE says "review-ish"
    svc.XGB = DummyXGB(p=0.50)
    svc.PREPROCESS = DummyPreprocess()
    svc.AE = DummyAE(err=0.0)
    svc.FEATURES = _minimal_features()
    svc.AE_THRESHOLDS = {"review": 1.0, "block": 2.0}

    monkeypatch.setattr(svc, "_reconstruction_error", lambda *args, **kwargs: 1.5)

    out = svc.predict_hybrid(_minimal_payload())
    assert out["label"] == "review"


def test_missing_field_raises(monkeypatch):
    svc.XGB = DummyXGB(p=0.50)
    svc.PREPROCESS = DummyPreprocess()
    svc.AE = DummyAE(err=0.0)
    svc.FEATURES = _minimal_features()
    svc.AE_THRESHOLDS = {"review": 1.0, "block": 2.0}

    monkeypatch.setattr(svc, "_reconstruction_error", lambda *args, **kwargs: 0.1)

    with pytest.raises(ValueError):
        svc.predict_hybrid({"f1": 1.0})  # missing f2, c1
