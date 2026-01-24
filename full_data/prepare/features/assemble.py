from typing import List, Dict, Any, Tuple
from collections.abc import MutableMapping
import numpy as _np
from .utils import one_hot, weighted_presence
from prepare.config.schema import (
    get_slot_size,
)
from shared.config import config

WeightedIndices = List[Tuple[int, float]]


def embedding_component(indices: WeightedIndices, size: int) -> List[float]:
    pr_conf = config["prompt_representation"]
    if not isinstance(pr_conf, (dict, MutableMapping)):
        raise RuntimeError("'prompt_representation' must be an object in config.json")
    if pr_conf.get("mode") != "embedding":
        return weighted_presence(indices, size)

    emb = _np.asarray(indices, dtype=_np.float32)
    if emb.ndim > 1:
        emb = emb.flatten()

    target_dim = size
    if "dim" in pr_conf:
        target_dim = int(pr_conf["dim"])
    vec = emb.tolist()[:target_dim]
    if len(vec) < target_dim:
        vec = vec + [0.0] * (target_dim - len(vec))
    vec = [float(x) for x in vec]
    mn = min(vec) if vec else 0.0
    if mn < 0.0:
        vec = [v - mn for v in vec]
    return vec


def get_component_vector(
    name: str,
    cfg_norm: float,
    steps_norm: float,
    lora_weight: float,
    sampler_idx: int,
    scheduler_idx: int,
    model_idx: int,
    lora_idx: int,
    width_norm: float,
    height_norm: float,
    aspect_ratio_norm: float,
    pos_indices: WeightedIndices,
    neg_indices: WeightedIndices,
    slots: Dict[str, Any],
    mode: str,
    dim: int,
) -> List[float]:
    if name == "cfg":
        return [cfg_norm]
    if name == "steps":
        return [steps_norm]
    if name == "lora_weight":
        return [lora_weight]
    if name == "width":
        return [width_norm]
    if name == "height":
        return [height_norm]
    if name == "aspect_ratio":
        return [aspect_ratio_norm]
    if name == "steps_cfg":
        return [steps_norm * cfg_norm]
    size = get_slot_size(name, slots, mode, dim)
    if name == "lora":
        return [float(x) for x in one_hot(lora_idx, size)]
    if name == "sampler":
        return [float(x) for x in one_hot(sampler_idx, size)]
    if name == "scheduler":
        return [float(x) for x in one_hot(scheduler_idx, size)]
    if name == "model":
        return [float(x) for x in one_hot(model_idx, size)]
    if name == "positive_terms":
        return embedding_component(pos_indices, size)
    if name == "negative_terms":
        return embedding_component(neg_indices, size)
    else:
        raise ValueError(f"Unknown component: {name}")


def assemble_feature_vector(
    cfg_norm: float,
    steps_norm: float,
    lora_weight: float,
    sampler_idx: int,
    scheduler_idx: int,
    model_idx: int,
    lora_idx: int,
    width_norm: float,
    height_norm: float,
    aspect_ratio_norm: float,
    pos_indices: WeightedIndices,
    neg_indices: WeightedIndices,
    slots: Dict[str, Any],
    norm: Dict[str, Any],
    entry: Dict[str, Any],
    sampler_status: str,
    scheduler_status: str,
    model_status: str,
    lora_status: str,
    pos_statuses: List[str],
    neg_statuses: List[str],
    order: List[str],
    mode: str,
    dim: int,
) -> Tuple[List[float], List[str]]:
    feature: List[float] = []
    for name in order:
        component = get_component_vector(
            name,
            cfg_norm,
            steps_norm,
            lora_weight,
            sampler_idx,
            scheduler_idx,
            model_idx,
            lora_idx,
            width_norm,
            height_norm,
            aspect_ratio_norm,
            pos_indices,
            neg_indices,
            slots,
            mode,
            dim,
        )
        feature.extend(component)
    overflowed: List[str] = []
    status_map = {
        "sampler": sampler_status,
        "scheduler": scheduler_status,
        "model": model_status,
        "lora": lora_status,
    }
    for name, st in status_map.items():
        if st == "overflow":
            overflowed.append(name)
    if any((s == "overflow" for s in pos_statuses)):
        overflowed.append("positive_terms")
    if any((s == "overflow" for s in neg_statuses)):
        overflowed.append("negative_terms")
    return (feature, overflowed)
