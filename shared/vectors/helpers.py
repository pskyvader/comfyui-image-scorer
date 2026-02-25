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
