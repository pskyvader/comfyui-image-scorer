from __future__ import annotations
from typing import List, Dict, Any
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

from .paths import filtered_data, interaction_data, prepare_config
from .helpers import one_hot, get_slot_size


def assemble_feature_vector(
    metadata: Dict[str, Any],
    pos_embedding: np.ndarray,
    neg_embedding: np.ndarray,
    category_indices: Dict[str, int],
) -> np.ndarray:
    """
    Assembles the metadata vector.
    If prepare_config_path is provided, it will be used to load schema/norm values.
    """
    # 1. Normalize scalars (strict: missing keys raise KeyError)
    _NORMALIZATION = prepare_config["normalization"]
    cfg = float(metadata["cfg"])
    cfg_norm = min(cfg / _NORMALIZATION["cfg_max"], 1.0)

    steps = float(metadata["steps"])
    steps_norm = min(steps / _NORMALIZATION["steps_max"], 1.0)

    lora_w = float(metadata["lora_weight"])

    width = float(metadata["width"])
    width_norm = min(width / _NORMALIZATION["width_max"], 1.0)

    height = float(metadata["height"])
    height_norm = min(height / _NORMALIZATION["width_max"], 1.0)

    ar = width / height if height > 0 else 1.0
    ar_norm = min(ar / _NORMALIZATION["aspect_ratio_max"], 1.0)

    # 2. Build parts
    vector_parts: List[float] = []
    _SCHEMA_ORDER = prepare_config["vector_schema"]["order"]

    for name in _SCHEMA_ORDER:
        part: List[float] = []
        if name == "cfg":
            part = [cfg_norm]
        elif name == "steps":
            part = [steps_norm]
        elif name == "lora_weight":
            part = [lora_w]
        elif name == "steps_cfg":
            part = [steps_norm * cfg_norm]
        elif name == "width":
            part = [width_norm]
        elif name == "height":
            part = [height_norm]
        elif name == "aspect_ratio":
            part = [ar_norm]
        elif name in ["lora", "sampler", "scheduler", "model"]:
            idx = category_indices[name]
            part = one_hot(idx, get_slot_size(name))

        elif name in ["positive_terms", "negative_terms"]:
            dim = get_slot_size(name)
            embedding = pos_embedding if name == "positive_terms" else neg_embedding
            emb = embedding.flatten()[:dim]

            if len(emb) < dim:
                emb = np.pad(emb, (0, dim - len(emb)))
                mn = np.min(emb)
                if mn < 0:
                    emb = emb - mn
            part = emb.tolist()

        vector_parts.extend(part)


    return np.array(vector_parts, dtype=np.float32)


def map_categorical_value(
    mapping: Dict[str, Dict[str, int]], category: str, value: str
) -> int:
    """Finds index for a value. Returns 0 if not found (assuming 0 is 'unknown')."""
    # Maps are structured as: {"ValueName": index, ...}

    cat_map = mapping.get(category, {})

    # Direct match
    if value in cat_map:
        return cat_map[value]

    # Case-insensitive match feature
    value_lower = value.lower()
    for k, v in cat_map.items():
        if k.lower() == value_lower:
            return v

    # Fallback to 0 (which corresponds to "unknown" in the loaded maps)
    return 0


def apply_feature_filter(vecs: List[np.ndarray]) -> List[np.ndarray]:
    """
    Applies the feature filter (kept_indices) from filtered_data_cache.npz to the input vector.
    model_bin_dir: directory containing filtered_data_cache.npz
    """
    data = np.load(filtered_data)
    kept_indices = data["kept_indices"]
    results: List[np.ndarray] = []
    for vec in vecs:
        filtered_vector = vec[kept_indices]
        results.append(filtered_vector)
    return results


def apply_interaction_features(vecs: List[np.ndarray]) -> np.ndarray:
    """
    Applies the interaction features (from interaction_data_cache.npz) to the input vector.
    model_bin_dir: directory containing interaction_data_cache.npz
    """
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
    vecs_np = np.array(vecs)  # shape: (batch_size, feature_dim)
    poly_features = poly.fit_transform(vecs_np)  # shape: (batch_size, n_poly_features)

    # Only keep the selected interaction features
    data = np.load(interaction_data)
    n_features_in = vecs_np.shape[1]
    interaction_indices = data["interaction_indices"]
    inter_feats = poly_features[:, n_features_in:][:, interaction_indices]

    # Concatenate original features and selected interaction features
    X_final = np.hstack([vecs_np, inter_feats])
    return X_final
