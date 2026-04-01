import numpy as np
import numpy.typing as npt
from typing import Any


def l2_normalize_batch(vectors: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    eps: float = 1e-12
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return vectors / norms


def get_value_from_entry(
    entry: dict[str, Any], name: str, alias: list[str] | None = None
) -> Any:
    custom_text: dict[str, Any] = entry["custom_text"] if "custom_text" in entry else {}
    # print(
    #     f"entry: {entry}, name: {name}, custom_text: {custom_text}, type: {type(entry)}",
    #     flush=True,
    # )
    custom_text.update(entry)
    # current_value = custom_text[name] if name in custom_text else None
    # if alias is not None and len(alias) > 0:
    #     for a in alias:
    #         if a in custom_text:
    #             current_value = custom_text[a]
    #             break
    # 1. Check if aliases exist (the 20% case)
    if alias:
        # Find the first valid alias, or fall back to the name
        current_value = next((custom_text[a] for a in alias if a in custom_text), custom_text.get(name))
    else:
        # 2. Fast path for the 80% case (no aliases)
        current_value = custom_text.get(name)
        
    
    # if current_value is None:
    #     print(f"Value not found for {name}, returning None")

    return current_value
