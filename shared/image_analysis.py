import time
import logging
import os
import numpy as np
import numpy.typing as npt
from tqdm import tqdm
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Callable
from typing import Any
from skimage.feature import local_binary_pattern

from .config import config
from .io import atomic_write_json
from .vectors.image_vector import ImageVector

logger = logging.getLogger(__name__)

# Type Alias for the shared data structure
ImageEntry = tuple[str, dict[str, Any], str, str]

processed_cache: dict[str, ImageEntry] = {}

REQUIRED_ANALYSIS_FIELDS: set[str] = {
    "original_width",
    "original_height",
    "final_width",
    "final_height",
    "final_aspect_ratio",
    "original_aspect_ratio",
    "contrast",
    "sharpness",
    "noise_score",
    "colorfulness",
    "artifact_score",
    "edge_density",
    "texture_lbp",
}


def process_single_batch(
    prepare_func: Callable[[list[str]], list[Image.Image]],
    analyze_func: Callable[[list[Image.Image], list[ImageEntry]], list[ImageEntry]],
    save_func: Callable[[ImageEntry], None],
    paths: list[str],
    data: list[ImageEntry],
) -> list[ImageEntry]:

    image_batch: list[Image.Image] = prepare_func(paths)
    result: list[ImageEntry] = analyze_func(image_batch, data)

    for entry in result:
        save_func(entry)

    return result


class ImageAnalysis(ImageVector):
    def __init__(self, raw_data: list[ImageEntry]) -> None:
        image_entries = [v for v in config["vector"]["vectors"] if v["type"] == "image"]
        if not image_entries:
            raise KeyError("No image-type entries found in vector_config")
        model_key = image_entries[0]["model_key"]
        super().__init__("tmp_image", model_key=model_key)
        self.raw_data: list[ImageEntry] = raw_data
        self.processed_data: list[ImageEntry] = []

        for data in self.raw_data:
            image_path = data[0]
            file_id = data[3]
            self.path_list[file_id] = image_path

        r, g, b = 255.0, 0.0, 0.0
        rg, yb = r - g, 0.5 * (r + g) - b
        self.max_colorfulness: float = float(
            np.sqrt(rg**2 + yb**2) + 0.3 * np.sqrt(rg**2 + yb**2)
        )

    @staticmethod
    def _entry_has_required_fields(entry: dict[str, Any]) -> bool:
        keys = set(entry.keys())
        if isinstance(entry.get("custom_text"), dict):
            keys.update(entry["custom_text"].keys())
        return REQUIRED_ANALYSIS_FIELDS.issubset(keys)

    @staticmethod
    def _entry_json_path(entry: ImageEntry) -> str:
        img_path = entry[0]
        return img_path.rsplit(".", 1)[0] + ".json"

    def _save_entry_sidecar(self, entry: ImageEntry) -> None:
        json_path = self._entry_json_path(entry)

        if not os.path.exists(json_path):
            logger.warning("Sidecar JSON not found, skipping save: %s", json_path)
            return

        atomic_write_json(json_path, entry[1], indent=2)

    def _image_size(
        self, img: Image.Image, entry: dict[str, Any], data: ImageEntry
    ) -> dict[str, Any]:
        size_keys = [
            "original_width",
            "original_height",
            "final_width",
            "final_height",
            "final_aspect_ratio",
            "original_aspect_ratio",
        ]
        if all(k in entry for k in size_keys):
            return entry
        _start = time.perf_counter()
        w, h = img.size
        if h < 128 or w < 128:
            raise ValueError(f"Image too small: {data[0]}")

        entry.update(
            {
                "original_width": entry["width"] if "width" in entry else w,
                "original_height": entry["height"] if "height" in entry else h,
                "final_width": w,
                "final_height": h,
                "final_aspect_ratio": round(w / h, 4) if h > 0 else 0.0,
            }
        )
        entry["original_aspect_ratio"] = (
            entry["aspect_ratio"]
            if "aspect_ratio" in entry
            else entry["final_aspect_ratio"]
        )

        return entry

    def _contrast(self, img: Image.Image, entry: dict[str, Any]) -> dict[str, Any]:
        if "contrast" in entry:
            return entry
        _start = time.perf_counter()
        rgb = np.asarray(img).astype(np.float64) / 255.0
        lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        entry["contrast"] = float(np.std(lum)) / 0.5

        return entry

    def _sharpness(self, img: Image.Image, entry: dict[str, Any]) -> dict[str, Any]:
        if "sharpness" in entry:
            return entry
        _start = time.perf_counter()
        gray: npt.NDArray[np.float32] = np.asarray(img.convert("L"), dtype=np.float32)
        lap: npt.NDArray[np.float32] = (
            -4 * gray
            + np.roll(gray, 1, 0)
            + np.roll(gray, -1, 0)
            + np.roll(gray, 1, 1)
            + np.roll(gray, -1, 1)
        )
        raw_sharp: float = float(np.var(lap))
        entry["sharpness"] = float(raw_sharp / (raw_sharp + 500.0))

        return entry

    def _noise_score(self, img: Image.Image, entry: dict[str, Any]) -> dict[str, Any]:
        if "noise_score" in entry:
            return entry
        _start = time.perf_counter()
        gray: npt.NDArray[np.float32] = np.asarray(img.convert("L"), dtype=np.float32)
        blur: npt.NDArray[np.float32] = (
            gray
            + np.roll(gray, 1, 0)
            + np.roll(gray, -1, 0)
            + np.roll(gray, 1, 1)
            + np.roll(gray, -1, 1)
        ) / 5.0
        res: npt.NDArray[np.float32] = gray - blur
        norm: float = np.log1p(float(np.var(res))) / np.log1p(65025.0)
        entry["noise_score"] = float(norm * 4.0)

        return entry

    def _colorfulness(self, img: Image.Image, entry: dict[str, Any]) -> dict[str, Any]:
        if "colorfulness" in entry:
            return entry
        _start = time.perf_counter()
        rgb: npt.NDArray[np.float32] = np.asarray(img, dtype=np.float32)
        rg, yb = (
            rgb[..., 0] - rgb[..., 1],
            0.5 * (rgb[..., 0] + rgb[..., 1]) - rgb[..., 2],
        )
        raw: float = float(
            np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2)
            + 0.3 * np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2)
        )
        entry["colorfulness"] = float(raw / self.max_colorfulness)

        return entry

    def _artifact_score(
        self, img: Image.Image, entry: dict[str, Any]
    ) -> dict[str, Any]:
        if "artifact_score" in entry:
            return entry
        _start = time.perf_counter()
        gray: npt.NDArray[np.float32] = np.asarray(img.convert("L"), dtype=np.float32)
        h, w = gray.shape
        sx, sy = max(4, min(8, w // 256)), max(4, min(8, h // 256))
        vx, hy = np.arange(sx, w, sx), np.arange(sy, h, sy)

        v = np.abs(gray[:, vx] - gray[:, vx - sx]).mean() if vx.size > 0 else 0.0
        h_val = np.abs(gray[hy, :] - gray[hy - sy, :]).mean() if hy.size > 0 else 0.0

        rng: float = float(gray.max() - gray.min()) or 1.0
        entry["artifact_score"] = float(min(1.0, (v + h_val) / (2.0 * rng) * 5.0))

        return entry

    def _edge_density(self, img: Image.Image, entry: dict[str, Any]) -> dict[str, Any]:
        if "edge_density" in entry:
            return entry
        _start = time.perf_counter()
        gray: npt.NDArray[np.float32] = np.asarray(img.convert("L"), dtype=np.float32)
        dx = np.roll(gray, -1, 1) - np.roll(gray, 1, 1)
        dy = np.roll(gray, -1, 0) - np.roll(gray, 1, 0)
        mag: npt.NDArray[np.float32] = np.sqrt(dx**2 + dy**2)
        entry["edge_density"] = float(np.mean(mag) / 255.0)

        return entry

    def _texture_lbp(self, img: Image.Image, entry: dict[str, Any]) -> dict[str, Any]:
        if "texture_lbp" in entry:
            return entry
        _start = time.perf_counter()
        gray: npt.NDArray[np.uint8] = np.asarray(img.convert("L"), dtype=np.uint8)
        lbp: npt.NDArray[np.float64] = local_binary_pattern(
            gray, 8, 1, method="uniform"
        )
        hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10), density=True)
        entropy: float = -np.sum(hist * np.log(hist + 1e-9))
        entry["texture_lbp"] = float(entropy / np.log(10.0))

        return entry

    def analyze_image_batch(
        self,
        image_batch: list[Image.Image],
        data_batch: list[ImageEntry],
    ) -> list[ImageEntry]:
        if len(image_batch) != len(data_batch):
            raise IndexError("Batch mismatch")

        for i, img in enumerate(image_batch):
            path, entry, cat, extra = data_batch[i]
            entry = self._artifact_score(img, entry)
            entry = self._colorfulness(img, entry)
            entry = self._contrast(img, entry)
            entry = self._image_size(img, entry, data_batch[i])
            entry = self._noise_score(img, entry)
            entry = self._sharpness(img, entry)
            entry = self._edge_density(img, entry)
            entry = self._texture_lbp(img, entry)
            data_batch[i] = (path, entry, cat, extra)

        return data_batch

    def analyze_images_from_paths(
        self, batch_size: int, max_workers: int
    ) -> list[ImageEntry]:
        global processed_cache
        new_path: list[str] = []
        new_raw: list[ImageEntry] = []
        result: list[ImageEntry] = []

        for id, path in self.path_list.items():
            current_raw: ImageEntry = next((d for d in self.raw_data if d[3] == id))

            if id in processed_cache:
                result.append(processed_cache[id])
                continue

            if self._entry_has_required_fields(current_raw[1]):
                result.append(current_raw)
                continue

            new_path.append(path)
            new_raw.append(current_raw)

        total: int = len(new_path)

        if total > 0:
            batches = [
                (
                    self.prepare_image_batch,
                    self.analyze_image_batch,
                    self._save_entry_sidecar,
                    (new_path)[i : i + batch_size],
                    new_raw[i : i + batch_size],
                )
                for i in range(0, total, batch_size)
            ]

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(process_single_batch, *b) for b in batches]
                with tqdm(total=total, desc="Analyzing", unit="img") as pbar:
                    for f in as_completed(futures):
                        res: list[ImageEntry] = f.result()
                        result.extend(res)
                        pbar.update(len(res))

        for entry in result:
            processed_cache[entry[3]] = entry

        self.processed_data = [
            entry
            for id, entry in list(processed_cache.items())
            if id in set(self.path_list.keys())
        ]

        return result
