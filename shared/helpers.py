import numpy as np
from pathlib import Path
from PIL import Image
import shutil
import torch
from torch import Tensor
from .paths import models_dir, vectors_dir


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
    if not pil_images:
        return torch.zeros((1, 1, 1, 3), dtype=torch.float32)  # type: ignore[union-attr]

    tensors: list[Tensor] = []

    for img in pil_images:
        np_img = np.array(img.convert("RGB"))
        np_img = np_img.astype(np.float32) / 255.0
        t = torch.from_numpy(np_img).unsqueeze(0)  # type: ignore[union-attr]
        tensors.append(t)

    return torch.cat(tensors, dim=0)  # type: ignore[union-attr]