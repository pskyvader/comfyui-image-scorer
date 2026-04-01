import numpy as np
from pathlib import Path
from PIL import Image
import shutil
from .paths import models_dir, vectors_dir
from torch import Tensor


def remove_directory(directory_path: Path) -> None:
    if directory_path.exists():
        print(f"Removing {directory_path}")
        try:
            shutil.rmtree(directory_path)
        except OSError as e:
            print(f"Warning: Could not remove {directory_path}: {e}")


def remove_vectors() -> None:
    directory_path = Path(vectors_dir)
    remove_directory(directory_path)


def remove_models() -> None:
    directory_path = Path(models_dir)
    remove_directory(directory_path)


def export_image_batch(pil_images: list[Image.Image]) -> Tensor:
    import torch
    if not pil_images:
        return torch.zeros((1, 1, 1, 3), dtype=torch.float32)

    tensors: list[Tensor] = []

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
