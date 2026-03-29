import numpy as np
import numpy.typing as npt
from tqdm import tqdm
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict, Any, Callable
from skimage.feature import local_binary_pattern

from .vectors.image_vector import ImageVector

# Type Alias for the shared data structure
ImageEntry = Tuple[str, Dict[str, Any], str, str]


def process_single_batch(
    prepare_func: Callable[[List[str]], List[Image.Image]],
    analyze_func: Callable[[List[Image.Image], List[ImageEntry]], List[ImageEntry]],
    paths: List[str],
    data: List[ImageEntry],
) -> List[ImageEntry]:
    image_batch: List[Image.Image] = prepare_func(paths)
    return analyze_func(image_batch, data)


class ImageAnalysis(ImageVector):
    def __init__(self, raw_data: List[ImageEntry]) -> None:
        super().__init__("tmp_image")
        self.raw_data: List[ImageEntry] = raw_data
        self.processed_data: List[ImageEntry] = []

        for data in self.raw_data:
            self.path_list.append(data[0])

        # Theoretical max colorfulness based on RGB boundaries
        r, g, b = 255.0, 0.0, 0.0
        rg, yb = r - g, 0.5 * (r + g) - b
        self.max_colorfulness: float = float(
            np.sqrt(rg**2 + yb**2) + 0.3 * np.sqrt(rg**2 + yb**2)
        )

    def _image_size(
        self, img: Image.Image, entry: Dict[str, Any], data: ImageEntry
    ) -> Dict[str, Any]:
        w, h = img.size
        if h < 128 or w < 128:
            raise ValueError(f"Image too small: {data[0]}")

        entry.update(
            {
                "final_width": w,
                "final_height": h,
                "final_aspect_ratio": round(w / h, 4) if h > 0 else 0.0,
                "original_width": entry.get("width", w),
                "original_height": entry.get("height", h),
            }
        )
        entry["original_aspect_ratio"] = entry.get(
            "aspect_ratio", entry["final_aspect_ratio"]
        )
        return entry

    def _contrast(self, img: Image.Image, entry: Dict[str, Any]) -> Dict[str, Any]:
        rgb: npt.NDArray[np.float32] = np.asarray(img, dtype=np.float32) / 255.0
        lum: npt.NDArray[np.float32] = (
            0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        )
        entry["contrast"] = float(np.std(lum)) / 0.5
        return entry

    def _sharpness(self, img: Image.Image, entry: Dict[str, Any]) -> Dict[str, Any]:
        gray: npt.NDArray[np.float32] = np.asarray(img.convert("L"), dtype=np.float32)
        lap: npt.NDArray[np.float32] = (
            -4 * gray
            + np.roll(gray, 1, 0)
            + np.roll(gray, -1, 0)
            + np.roll(gray, 1, 1)
            + np.roll(gray, -1, 1)
        )
        # [FIXED] Soft-clipping normalization (maps 500 var to ~0.5)
        raw_sharp: float = float(np.var(lap))
        entry["sharpness"] = float(raw_sharp / (raw_sharp + 500.0))
        return entry

    def _noise_score(self, img: Image.Image, entry: Dict[str, Any]) -> Dict[str, Any]:
        gray: npt.NDArray[np.float32] = np.asarray(img.convert("L"), dtype=np.float32)
        blur: npt.NDArray[np.float32] = (
            gray
            + np.roll(gray, 1, 0)
            + np.roll(gray, -1, 0)
            + np.roll(gray, 1, 1)
            + np.roll(gray, -1, 1)
        ) / 5.0
        # [FIXED] Log-scaled normalization for human sensitivity
        res: npt.NDArray[np.float32] = gray - blur
        norm: float = np.log1p(float(np.var(res))) / np.log1p(65025.0)
        entry["noise_score"] = float(norm * 4.0)
        return entry

    def _colorfulness(self, img: Image.Image, entry: Dict[str, Any]) -> Dict[str, Any]:
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
        self, img: Image.Image, entry: Dict[str, Any]
    ) -> Dict[str, Any]:
        gray: npt.NDArray[np.float32] = np.asarray(img.convert("L"), dtype=np.float32)
        h, w = gray.shape
        sx, sy = max(4, min(8, w // 256)), max(4, min(8, h // 256))
        vx, hy = np.arange(sx, w, sx), np.arange(sy, h, sy)

        v = np.abs(gray[:, vx] - gray[:, vx - sx]).mean() if vx.size > 0 else 0.0
        h_val = np.abs(gray[hy, :] - gray[hy - sy, :]).mean() if hy.size > 0 else 0.0

        # [FIXED] Intensity-normalized and sensitivity boosted
        rng: float = float(gray.max() - gray.min()) or 1.0
        entry["artifact_score"] = float(min(1.0, (v + h_val) / (2.0 * rng) * 5.0))
        return entry

    def _edge_density(self, img: Image.Image, entry: Dict[str, Any]) -> Dict[str, Any]:
        gray: npt.NDArray[np.float32] = np.asarray(img.convert("L"), dtype=np.float32)
        dx = np.roll(gray, -1, 1) - np.roll(gray, 1, 1)
        dy = np.roll(gray, -1, 0) - np.roll(gray, 1, 0)
        # [FIXED] Resolution-independent density (mean gradient magnitude)
        mag: npt.NDArray[np.float32] = np.sqrt(dx**2 + dy**2)
        entry["edge_density"] = float(np.mean(mag) / 255.0)
        return entry

    def _texture_lbp(self, img: Image.Image, entry: Dict[str, Any]) -> Dict[str, Any]:
        gray: npt.NDArray[np.uint8] = np.asarray(img.convert("L"), dtype=np.uint8)
        lbp: npt.NDArray[np.float64] = local_binary_pattern(
            gray, 8, 1, method="uniform"
        )
        hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10), density=True)
        # [FIXED] Entropy normalized by log(n_bins)
        entropy: float = -np.sum(hist * np.log(hist + 1e-9))
        entry["texture_lbp"] = float(entropy / np.log(10.0))
        return entry

    def analyze_image_batch(
        self,
        image_batch: List[Image.Image],
        data_batch: List[ImageEntry],
    ) -> List[ImageEntry]:
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
        self, batch_size: int = 32, max_workers: int = 16
    ) -> List[ImageEntry]:
        total: int = len(self.path_list)
        if total == 0:
            return []

        batches = [
            (
                self.prepare_image_batch,
                self.analyze_image_batch,
                self.path_list[i : i + batch_size],
                self.raw_data[i : i + batch_size],
            )
            for i in range(0, total, batch_size)
        ]

        results: List[ImageEntry] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_single_batch, *b) for b in batches]
            with tqdm(total=total, desc="Analyzing", unit="img") as pbar:
                for f in as_completed(futures):
                    res = f.result()
                    results.extend(res)
                    pbar.update(len(res))

        self.processed_data = results
        return self.processed_data
