from __future__ import annotations
from typing import Dict, Any

def get_slot_size(name: str, slots: Dict[str, Any], mode: str, dim: int) -> int:
    if name in ("positive_terms", "negative_terms") and mode == "embedding":
        return dim
    return slots[name] if name in slots else 0



IMAGE_VEC_LEN = 768