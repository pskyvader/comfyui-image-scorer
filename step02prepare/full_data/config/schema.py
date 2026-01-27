from __future__ import annotations
from typing import List, NamedTuple, Dict, Any
from step02prepare.full_data.config.manager import load_vector_schema


class ComponentInfo(NamedTuple):
    name: str
    offset: int
    size: int


def get_vector_order() -> List[str]:
    schema = load_vector_schema()
    order = schema.get("order")
    if not order:
        msg = "Schema 'order' array is missing or empty in config.json"
        raise ValueError(msg)
    return order


def get_slot_size(name: str, slots: Dict[str, Any], mode: str, dim: int) -> int:
    if name in ("positive_terms", "negative_terms") and mode == "embedding":
        return dim
    return slots[name] if name in slots else 0



IMAGE_VEC_LEN = 768


def get_feature_vector_length(
    slots: Dict[str, Any],
    mode: str,
    dim: int,
) -> int:
    return sum(get_slot_size(name, slots, mode, dim) for name in get_vector_order())


def get_total_vector_length(
    slots: Dict[str, Any], mode: str, dim: int, image_vec_len: int = IMAGE_VEC_LEN
) -> int:
    return image_vec_len + get_feature_vector_length(slots, mode, dim)
