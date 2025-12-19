from __future__ import annotations

import json
import logging
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, List, Union

logger = logging.getLogger(__name__)


def latency_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def _read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path.as_posix()}")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=8)
def load_required_raw_features(path: Union[str, Path] = "artifacts/features.json") -> List[str]:
    """
    Supports two shapes:
    1) list[str] -> treated as feature list
    2) dict      -> derives required raw features from:
       categorical_features + numerical_features
    """
    p = Path(path)
    data = _read_json(p)

    if isinstance(data, list) and all(isinstance(x, str) for x in data):
        return data

    if isinstance(data, dict):
        cat = data.get("categorical_features")
        num = data.get("numerical_features")

        if isinstance(cat, list) and isinstance(num, list) and all(isinstance(x, str) for x in (cat + num)):
            return list(cat) + list(num)

        for key in ("raw_features", "feature_order", "input_features"):
            val = data.get(key)
            if isinstance(val, list) and all(isinstance(x, str) for x in val):
                return val

    raise ValueError(
        f"{p.as_posix()} must be either a JSON list[str] or a dict with "
        f"'categorical_features' and 'numerical_features' (list[str])"
    )
