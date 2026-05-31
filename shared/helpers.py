import logging
import numpy as np
import shutil
import time
from pathlib import Path
from PIL import Image
import torch
from torch import Tensor
from .paths import models_dir, vectors_dir

logger = logging.getLogger(__name__)


def remove_directory(directory_path: Path) -> None:
    _start = time.perf_counter()
    if directory_path.exists():
        logger.info(f"Removing {directory_path}")
        shutil.rmtree(directory_path)


def remove_vectors() -> None:
    _start = time.perf_counter()
    directory_path = Path(vectors_dir)
    remove_directory(directory_path)


def remove_models() -> None:
    _start = time.perf_counter()
    directory_path = Path(models_dir)
    remove_directory(directory_path)


def export_image_batch(pil_images: list[Image.Image]) -> Tensor:
    _start = time.perf_counter()
    if not pil_images:
        result = torch.zeros((1, 1, 1, 3), dtype=torch.float32)  # type: ignore[reportPrivateImportUsage]
    else:
        tensors: list[Tensor] = []
        for img in pil_images:
            np_img = np.array(img.convert("RGB"))
            np_img = np_img.astype(np.float32) / 255.0
            t = torch.from_numpy(np_img).unsqueeze(0)  # type: ignore[reportPrivateImportUsage]
            tensors.append(t)
        result = torch.cat(tensors, dim=0)  # type: ignore[reportPrivateImportUsage]

    return result
