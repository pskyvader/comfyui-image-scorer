from __future__ import annotations
from typing import List, Dict, Any, cast, Optional
import numpy as np
import os
import json


def _set_defaults_from_prepare_config(data: Dict[str, Any]):
    """Apply values from parsed prepare_config.json to module state."""
    global _SCHEMA_ORDER, _SLOT_SIZES, _NORMALIZATION, _EMBEDDING_DIM

    # Vector schema
    vs = data.get("vector_schema")
    order = vs.get("order")
    slots = vs.get("slots")

    if isinstance(order, list) and order:
        _SCHEMA_ORDER = order
    if isinstance(slots, dict) and slots:
        _SLOT_SIZES = {k: int(v) for k, v in slots.items()}

    # normalization
    norm = data.get("normalization")
    if isinstance(norm, dict) and norm:
        _NORMALIZATION = {k: float(v) for k, v in norm.items()}

    # prompt embedding dim
    pr = data.get("prompt_representation") or {}
    dim = pr.get("dim")
    if isinstance(dim, int) and dim > 0:
        _EMBEDDING_DIM = dim


def load_prepare_config(path: Optional[str] = None) -> Optional[str]:
    """Loads the prepare_config.json and sets module constants.

    If `path` is provided and the file is missing, a `FileNotFoundError` is raised.
    If `path` is None the function will attempt to discover the config file by
    checking common locations (node root, repository root). Returns the path
    that was loaded or None if no config was found (when path is None).
    """
    global _LOADED_CONFIG_PATH

    # If explicit path provided, fail fast if missing to enforce strict configuration
    if path:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Prepare config not found at provided path: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        _set_defaults_from_prepare_config(data)
        _LOADED_CONFIG_PATH = path
        return path

    # Auto-discovery when no explicit path given
    candidates = []

    # 1) Node-local file (if this module is inside the node package)
    this_dir = os.path.dirname(__file__)
    node_root = os.path.abspath(os.path.join(this_dir, ".."))
    candidates.append(os.path.join(node_root, "prepare_config.json"))

    # 2) Common dev repo location (two levels up from node lib)
    repo_root = os.path.abspath(os.path.join(this_dir, "..", ".."))

    # Prefer consolidated config folder
    candidates.append(os.path.join(repo_root, "config", "prepare_config.json"))
    # Optional node-specific override in config folder
    candidates.append(os.path.join(repo_root, "config", "comfy_prepare_config.json"))
    # Legacy location (repo root)
    candidates.append(os.path.join(repo_root, "prepare_config.json"))

    for c in candidates:
        if c and os.path.exists(c):
            try:
                with open(c, "r", encoding="utf-8") as f:
                    data = json.load(f)
                _set_defaults_from_prepare_config(data)
                _LOADED_CONFIG_PATH = c
                return c
            except Exception:
                # If parsing fails, continue to next candidate
                continue
    return None


def one_hot(idx: int, size: int) -> List[float]:
    vec = [0.0] * size
    if 0 <= idx < size:
        vec[idx] = 1.0
    return vec


def get_slot_size(name: str) -> int:
    base = int(_SLOT_SIZES.get(name, 1))
    # Embedding slots result in a vector of length `dim` (from config)
    if name in ["positive_terms", "negative_terms"]:
        return int(_EMBEDDING_DIM)
    return base


def assemble_feature_vector(
    metadata: Dict[str, Any],
    pos_embedding: np.ndarray,
    neg_embedding: np.ndarray,
    category_indices: Dict[str, int],
    prepare_config_path: Optional[str] = None,
) -> 'np.ndarray':
    if np is None:
        raise ImportError("The 'numpy' package is required to use 'assemble_feature_vector'. Please install it or add it to your environment.")
    """
    Assembles the metadata vector.
    If prepare_config_path is provided, it will be used to load schema/norm values.
    """
    # Ensure configuration loaded (either auto-discovered or via provided path)
    if prepare_config_path:
        load_prepare_config(prepare_config_path)
    else:
        # Try to autodiscover at import time if not already loaded
        if _LOADED_CONFIG_PATH is None:
            load_prepare_config()

    # 1. Normalize scalars (strict: missing keys raise KeyError)
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
        elif name == "lora":
            idx = category_indices.get("lora", 0)
            part = one_hot(idx, get_slot_size("lora"))
        elif name == "sampler":
            idx = category_indices.get("sampler", 0)
            part = one_hot(idx, get_slot_size("sampler"))
        elif name == "scheduler":
            idx = category_indices.get("scheduler", 0)
            part = one_hot(idx, get_slot_size("scheduler"))
        elif name == "model":
            idx = category_indices.get("model", 0)
            part = one_hot(idx, get_slot_size("model"))
        elif name == "positive_terms":
            # Embedding handling - use configured embedding dimension
            dim = _EMBEDDING_DIM
            emb = pos_embedding.flatten()[:dim]
            if len(emb) < dim:
                emb = np.pad(emb, (0, dim - len(emb)))

            # Shift to positive if needed
            mn = np.min(emb)
            if mn < 0:
                emb = emb - mn
            part = emb.tolist()

        elif name == "negative_terms":
            dim = _EMBEDDING_DIM
            emb = neg_embedding.flatten()[:dim]
            if len(emb) < dim:
                emb = np.pad(emb, (0, dim - len(emb)))
            mn = np.min(emb)
            if mn < 0:
                emb = emb - mn
            part = emb.tolist()

        vector_parts.extend(part)

    return np.array(vector_parts, dtype=np.float32)


def load_maps(maps_path: str) -> Dict[str, Dict[str, int]]:
    """
    Loads categorical maps from json files in maps_path.
    Expected format in JSON is a LIST of strings: ["unknown", "name1", "name2", ...]
    We convert this to a Dict: {"unknown": 0, "name1": 1, "name2": 2, ...}
    """
    import json
    import os

    mapping: Dict[str, Dict[str, int]] = {}
    for name in ["sampler", "scheduler", "model", "lora"]:
        fname = os.path.join(maps_path, f"{name}_map.json")
        if os.path.exists(fname):
            with open(fname, "r") as f:
                # Based on file content, these are LISTS, not DICTS.
                data = json.load(f)
                if isinstance(data, list):
                    # Convert list ["val0", "val1"] -> {"val0": 0, "val1": 1}
                    data_list = cast(List[str], data)
                    mapping[name] = {v: k for k, v in enumerate(data_list)}
                else:
                    mapping[name] = cast(Dict[str, int], data)
        else:
            raise FileNotFoundError(f"Map not found: {fname}")
    return mapping


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


# --- Feature Filtering and Interaction Application (for ComfyUI node) ---
import os


def apply_feature_filter(
    vecs: List[np.ndarray], model_bin_dir: str
) -> List[np.ndarray]:
    """
    Applies the feature filter (kept_indices) from filtered_data_cache.npz to the input vector.
    model_bin_dir: directory containing filtered_data_cache.npz
    """
    cache_path = os.path.join(model_bin_dir, "filtered_data_cache.npz")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Feature filter cache not found: {cache_path}")
    data = np.load(cache_path)
    kept_indices = data["kept_indices"]
    results: List[np.ndarray] = []
    print(f"Applying feature filter, keeping {len(kept_indices)} features.")
    print(f"Kept indices: {kept_indices}")
    for vec in vecs:
        print(f"vector: {vec}.Applying filter to vector...")
        print(f"Original vector length: {len(vec)}")
        print(f"Filtered vector length: {len(kept_indices)}")
        results.append(vec[kept_indices])
    return results


def apply_interaction_features(
    vecs: List[np.ndarray], model_bin_dir: str
) -> np.ndarray:
    """
    Applies the interaction features (from interaction_data_cache.npz) to the input vector.
    model_bin_dir: directory containing interaction_data_cache.npz
    """
    cache_path = os.path.join(model_bin_dir, "interaction_data_cache.npz")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Interaction cache not found: {cache_path}")
    data = np.load(cache_path)
    interaction_indices = data["interaction_indices"]
    # Rebuild the interaction features for this single vector
    # PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
    from sklearn.preprocessing import PolynomialFeatures

    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
    # poly.fit_transform expects 2D array
    vecs_np = np.array(vecs)  # shape: (batch_size, feature_dim)
    n_features_in = vecs_np.shape[1]
    poly_features = poly.fit_transform(vecs_np)  # shape: (batch_size, n_poly_features)
    # Only keep the selected interaction features
    inter_feats = poly_features[:, n_features_in:][:, interaction_indices]
    # Concatenate original features and selected interaction features
    X_final = np.hstack([vecs_np, inter_feats])
    return X_final


# Public aliases for backward compatibility and tests
SCHEMA_ORDER = _SCHEMA_ORDER
SLOT_SIZES = _SLOT_SIZES
EMBEDDING_DIM = _EMBEDDING_DIM
