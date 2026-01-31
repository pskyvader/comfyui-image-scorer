import numpy as np
from pathlib import Path
from typing import Dict, Any, cast, List
from PIL import Image
import torch
import pickle
import base64
import json
import os
from .paths import maps_dir, prepare_config


def one_hot(idx: int, size: int) -> List[float]:
    vec = [0.0] * size
    if 0 <= idx < size:
        vec[idx] = 1.0
    return vec


def get_slot_size(name: str) -> int:
    _SLOT_SIZES = prepare_config["vector_schema"]["slots"]
    base = int(_SLOT_SIZES.get(name, 1))
    # Embedding slots result in a vector of length `dim` (from config)
    if name in ["positive_terms", "negative_terms"]:
        return int(prepare_config["prompt_representation"]["dim"])
    return base


def load_maps() -> Dict[str, Dict[str, int]]:
    """
    Loads categorical maps from json files in maps_path.
    Expected format in JSON is a LIST of strings: ["unknown", "name1", "name2", ...]
    We convert this to a Dict: {"unknown": 0, "name1": 1, "name2": 2, ...}
    """


    mapping: Dict[str, Dict[str, int]] = {}
    for name in ["sampler", "scheduler", "model", "lora"]:
        fname = os.path.join(maps_dir, f"{name}_map.json")
        if os.path.exists(fname):
            with open(fname, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    data_list = cast(List[str], data)
                    mapping[name] = {v: k for k, v in enumerate(data_list)}
                else:
                    mapping[name] = cast(Dict[str, int], data)
        else:
            raise FileNotFoundError(f"Map not found: {fname}")
    return mapping


def load_model(model_path: str) -> Any:
    """Load the trained model from .npz file.

    Args:
        model_path: Path to the .npz file containing the model

    Returns:
        The trained model object

    Raises:
        KeyError: If '__model_b64__' key is not found in the .npz file
    """


    with np.load(Path(model_path), allow_pickle=True) as npz:
        if "__model_b64__" not in npz.files:
            raise KeyError(
                f"No '__model_b64__' key found in {model_path}. Available keys: {list(npz.files)}"
            )
        model_b64 = npz["__model_b64__"].item()
        model_bytes = base64.b64decode(model_b64.encode("ascii"))
        return pickle.loads(model_bytes)


def _normalize(val: Any) -> Any:
    if isinstance(val, np.ndarray):
        if val.shape == ():
            return val.item()
        return val.copy()
    return val


def load_model_diagnostics(model_path: str) -> Dict[str, Any]:
    with np.load(Path(model_path), allow_pickle=True) as npz:
        return {k: _normalize(npz[k]) for k in npz.files}


def get_param(key, data):
    if key in data:
        return data[key]
    if "metrics" in data:
        m = data["metrics"]
        if hasattr(m, "item"):
            m = m.item()
        if isinstance(m, dict) and key in m:
            return m[key]
    return None


def prepare_image_batch(image: Any) -> list[Image.Image]:
    """
    Convert various image inputs into a list of RGB PIL Images.
    Accepts torch.Tensor, numpy arrays, PIL.Image, or list/tuple of these.
    Returns a list of `PIL.Image.Image` (RGB).
    """

    def array_to_pil(arr: Any) -> List[Image.Image]:
        arr = np.asarray(arr)
        if arr.ndim == 4:
            # Batch of images
            out: List[Image.Image] = []
            for i in range(arr.shape[0]):
                sub = array_to_pil(arr[i])
                out.extend(sub)
            return out
        if arr.ndim == 3:
            # Heuristic: if first dim is small and likely channels -> (C,H,W)
            if arr.shape[0] in (1, 3, 4):
                # Treat as C,H,W -> convert to H,W,C
                arr = np.transpose(arr, (1, 2, 0))
            # Now arr should be H,W,C
            if arr.shape[2] == 1:
                arr = np.repeat(arr, 3, axis=2)
            if arr.shape[2] == 4:
                arr = arr[:, :, :3]
            if np.issubdtype(arr.dtype, np.floating):
                arr = np.clip(arr, 0.0, 1.0)
                arr = (arr * 255.0).round().astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            return [Image.fromarray(arr).convert("RGB")]
        if arr.ndim == 2:
            # Grayscale
            if np.issubdtype(arr.dtype, np.floating):
                arr = np.clip(arr, 0.0, 1.0)
                arr = (arr * 255.0).round().astype(np.uint8)
            arr = np.stack([arr] * 3, axis=-1)
            return [Image.fromarray(arr).convert("RGB")]
        raise ValueError(f"Unsupported ndarray shape: {arr.shape}")

    # Start handling different input types
    if isinstance(image, Image.Image):
        return [image.convert("RGB")]
    if isinstance(image, torch.Tensor):
        res = array_to_pil(image.detach().cpu().numpy())
        return res
    if isinstance(image, np.ndarray):
        res = array_to_pil(image)
        return res
    if isinstance(image, (list, tuple)):
        out: List[Image.Image] = []
        for it in image:
            out.extend(prepare_image_batch(it))
        return out
    raise TypeError(f"Unsupported image type: {type(image)}")


def export_image_batch(pil_images: List[Image.Image]) -> torch.Tensor:
    if not pil_images:
        return torch.zeros((1, 1, 1, 3), dtype=torch.float32)

    tensors: List[torch.Tensor] = []

    for img in pil_images:
        # 1. Ensure RGB and convert to NumPy array (Shape: H, W, C)
        # ComfyUI requires HWC format, which is the default for PIL -> NumPy
        np_img = np.array(img.convert("RGB"))

        # 2. Normalize and convert to float32
        # ComfyUI expects values between 0.0 and 1.0
        np_img = np_img.astype(np.float32) / 255.0

        # 3. Convert to Torch Tensor and add batch dimension [1, H, W, C]
        t = torch.from_numpy(np_img).unsqueeze(0)
        tensors.append(t)

    # 4. Concatenate into a single batch [B, H, W, C]
    return torch.cat(tensors, dim=0)
