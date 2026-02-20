import numpy as np
from typing import Any, List, Tuple, Dict
from PIL import Image
import torch


def get_value_from_entry(entry: Dict[str, Any], name: str):
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


def load_clip() -> Tuple[Any, Any]:
    global _clip_model, _clip_processor
    if _clip_model is not None:
        return
    device = config["vision_model"]["device"]
    VISION_MODEL_ID = config["vision_model"]["name"]
    if device != "cuda":
        raise RuntimeError("`clip_device` not set to 'cuda'")

    try:
        print(f"Loading Vision Model: {VISION_MODEL_ID}...")
        model = AutoModel.from_pretrained(VISION_MODEL_ID, local_files_only=True)
        processor = AutoProcessor.from_pretrained(
            VISION_MODEL_ID, local_files_only=True, use_fast=True
        )
    except Exception:
        # Fallback to online if not in cache (and if internet is available)
        print("Vision model not found in cache, attempting download...")
        model = AutoModel.from_pretrained(VISION_MODEL_ID)
        processor = AutoProcessor.from_pretrained(VISION_MODEL_ID)
    model = model.eval()
    model.to(device)
    _clip_model = model
    _clip_processor = processor
    print(f"Vision model loaded on device: {device} ")


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
