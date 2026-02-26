import numpy as np
from typing import Any, List, Dict
from PIL import Image
import torch


def l2_normalize_batch(vectors: np.ndarray) -> np.ndarray:
    eps: float = 1e-12
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return vectors / norms


def get_value_from_entry(entry: Dict[str, Any], name: str) -> Any:
    custom_text: Dict[str, Any] = entry["custom_text"] if "custom_text" in entry else {}
    # print(
    #     f"entry: {entry}, name: {name}, custom_text: {custom_text}, type: {type(entry)}",
    #     flush=True,
    # )
    current_value = (
        entry[name]
        if name in entry
        else (custom_text[name] if name in custom_text else None)
    )

    # if current_value is None:
    #     print(f"Value not found for {name}, returning None")

    return current_value
