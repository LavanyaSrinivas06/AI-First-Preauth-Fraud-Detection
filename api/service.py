from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Union

import joblib
import pandas as pd

from api.utils import load_required_raw_features

logger = logging.getLogger(__name__)

DEFAULT_PREPROCESS_PATH = Path("artifacts/preprocess.joblib")
DEFAULT_FEATURES_PATH = Path("artifacts/features.json")


class Preprocessor(Protocol):
    def transform(self, X: pd.DataFrame) -> Any: ...


@dataclass(frozen=True)
class Artifacts:
    preprocess: Preprocessor
    required_raw_features: List[str]


_lock = threading.Lock()
_ARTIFACTS: Optional[Artifacts] = None

FEATURE_ALIASES = {
    "amount": "Amount",  # pipeline expects "Amount"
}


def _apply_aliases(payload: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(payload)
    for api_key, train_key in FEATURE_ALIASES.items():
        if api_key in out and train_key not in out:
            out[train_key] = out[api_key]
    return out


def load_artifacts(
    preprocess_path: Union[str, Path] = DEFAULT_PREPROCESS_PATH,
    features_path: Union[str, Path] = DEFAULT_FEATURES_PATH,
) -> Artifacts:
    global _ARTIFACTS
    with _lock:
        if _ARTIFACTS is not None:
            return _ARTIFACTS

        p = Path(preprocess_path)
        if not p.exists():
            raise FileNotFoundError(f"Missing preprocessing artifact: {p.as_posix()}")

        preprocess = joblib.load(p)
        if not hasattr(preprocess, "transform"):
            raise TypeError("Loaded preprocess artifact does not implement `transform`")

        required_raw_features = load_required_raw_features(features_path)

        _ARTIFACTS = Artifacts(preprocess=preprocess, required_raw_features=required_raw_features)
        logger.info("Artifacts loaded: preprocess=%s, required_raw_features=%d", p.as_posix(), len(required_raw_features))
        return _ARTIFACTS


def unload_artifacts() -> None:
    global _ARTIFACTS
    with _lock:
        _ARTIFACTS = None


def is_ready() -> bool:
    return _ARTIFACTS is not None


def build_dataframe(payload: Dict[str, Any], required_raw_features: List[str]) -> pd.DataFrame:
    payload = _apply_aliases(payload)
    row = {f: payload.get(f, None) for f in required_raw_features}
    return pd.DataFrame([row], columns=required_raw_features)


def predict_placeholder() -> Dict[str, Any]:
    return {
        "label": "review",
        "score_xgb": None,
        "score_ae": None,
        "ensemble_score": None,
        "reason_codes": [],
    }


def run_inference(payload: Dict[str, Any]) -> Dict[str, Any]:
    artifacts = load_artifacts()
    df = build_dataframe(payload, artifacts.required_raw_features)

    try:
        _ = artifacts.preprocess.transform(df)
    except Exception as exc:
        logger.exception("Preprocessing failed during inference")
        raise RuntimeError("Preprocessing failed") from exc

    return predict_placeholder()
