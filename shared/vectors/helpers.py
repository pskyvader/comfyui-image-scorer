from __future__ import annotations

import time
from typing import Any

import numpy as np
import numpy.typing as npt


def l2_normalize_batch(vectors: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    _start = time.perf_counter()
    eps: float = 1e-12
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    result = vectors / norms

    return result


def get_value_from_entry(
    entry: dict[str, Any], name: str, alias: list[str] | None = None
) -> Any:
    _start = time.perf_counter()
    custom_text: dict[str, Any] = {}
    if "custom_text" in entry and isinstance(entry["custom_text"], dict):
        custom_text.update(entry["custom_text"])
    for key, value in entry.items():
        custom_text[key] = value

    result: Any = None
    if alias is not None and len(alias) > 0:
        for alias_name in alias:
            if alias_name in custom_text:
                result = custom_text[alias_name]
                break
        if result is None and name in custom_text:
            result = custom_text[name]
    elif name in custom_text:
        result = custom_text[name]

    return result
