from typing import List, Tuple, Dict, Any, Iterable, Generator
from tqdm import tqdm
from PIL import Image
import numpy as np
from joblib import Parallel, delayed
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Dict, Any
from tqdm import tqdm

from .vectors.image_vector import ImageVector




from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict, Any, Optional, Set
from tqdm import tqdm
import os




def process_single_batch(
    prepare_func: Any,  # Technically Callable[..., List[Image.Image]]
    analyze_func: Any,  # Technically Callable[..., List[AnalysisResult]]
    paths: List[str],
    data: List[Tuple[str, Dict[str, Any], str, str]],
) -> List[Tuple[str, Dict[str, Any], str, str]]:
    """
    Standalone wrapper to process one batch.
    Logic: paths -> prepare -> images -> analyze -> results
    """
    # 1. Convert paths to PIL Image objects
    image_batch: List[Image.Image] = prepare_func(paths)

    # 2. Analyze the images and return the batch results
    processed_batch: List[Tuple[str, Dict[str, Any], str, str]] = analyze_func(
        image_batch, data
    )

    return processed_batch


class ImageAnalysis(ImageVector):
    def __init__(self, raw_data: List[Tuple[str, Dict[str, Any], str, str]]) -> None:
        super().__init__("tmp_image")
        self.raw_data = raw_data
        self.image_list: List[Image.Image]
        self.entries: List[Dict[str, Any]]
        self.processed_data: List[Tuple[str, Dict[str, Any], str, str]] = []

        for data in self.raw_data:
            (img_path, _, _, _) = data
            self.path_list.append(img_path)

    def _image_size(self, image: Image.Image, entry: Dict[str, Any]) -> Dict[str, Any]:
        w, h = image.size
        entry["final_width"] = w
        entry["final_height"] = h
        entry["final_aspect_ratio"] = round(w / h, 4) if h > 0 else 0.0

        entry["original_width"] = (
            entry["width"] if "width" in entry else entry["final_width"]
        )
        entry["original_height"] = (
            entry["height"] if "height" in entry else entry["final_height"]
        )
        entry["original_aspect_ratio"] = (
            entry["aspect_ratio"]
            if "aspect_ratio" in entry
            else entry["final_aspect_ratio"]
        )

        return entry

    def _contrast(self, img: Image.Image, entry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Contrast = standard deviation of luminance.
        Higher = more contrast.
        """
        rgb: np.ndarray = np.asarray(img, dtype=np.float32) / 255.0
        # Convert to luminance (ITU-R BT.601)
        luminance: np.ndarray = (
            0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
        )

        entry["contrast"] = float(np.std(luminance))
        return entry

    def _sharpness(self, img: Image.Image, entry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sharpness = variance of Laplacian.
        Higher = sharper image.
        """
        gray: np.ndarray = np.asarray(img.convert("L"), dtype=np.float32)

        laplacian: np.ndarray = (
            -4 * gray
            + np.roll(gray, 1, axis=0)
            + np.roll(gray, -1, axis=0)
            + np.roll(gray, 1, axis=1)
            + np.roll(gray, -1, axis=1)
        )

        entry["sharpness"] = float(np.var(laplacian))
        return entry

    def _noise_score(self, img: Image.Image, entry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Noise score = variance of high-frequency residual.
        Higher = more noise.
        """
        gray: np.ndarray = np.asarray(img.convert("L"), dtype=np.float32)

        blurred: np.ndarray = (
            gray
            + np.roll(gray, 1, axis=0)
            + np.roll(gray, -1, axis=0)
            + np.roll(gray, 1, axis=1)
            + np.roll(gray, -1, axis=1)
        ) / 5.0

        residual: np.ndarray = gray - blurred

        entry["noise_score"] = float(np.var(residual))
        return entry

    def _colorfulness(self, img: Image.Image, entry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Colorfulness metric (Hasler & Süsstrunk).
        Higher = more vivid color.
        """
        rgb: np.ndarray = np.asarray(img, dtype=np.float32)

        r: np.ndarray = rgb[:, :, 0]
        g: np.ndarray = rgb[:, :, 1]
        b: np.ndarray = rgb[:, :, 2]

        rg: np.ndarray = r - g
        yb: np.ndarray = 0.5 * (r + g) - b

        std_rg: float = float(np.std(rg))
        std_yb: float = float(np.std(yb))

        mean_rg: float = float(np.mean(rg))
        mean_yb: float = float(np.mean(yb))

        entry["colorfulness"] = float(
            np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2)
        )
        return entry

    def _artifact_score2(
        self, img: Image.Image, entry: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Simple JPEG blockiness detector.
        Measures discontinuities at 8px boundaries.
        Higher = more block artifacts.
        """
        gray: np.ndarray = np.asarray(img.convert("L"), dtype=np.float32)
        h: int
        w: int
        h, w = gray.shape

        vertical_edges: float = 0.0
        horizontal_edges: float = 0.0

        for x in range(8, w, 8):
            vertical_edges += float(np.mean(np.abs(gray[:, x] - gray[:, x - 1])))

        for y in range(8, h, 8):
            horizontal_edges += float(np.mean(np.abs(gray[y, :] - gray[y - 1, :])))

        entry["artifact_score"] = float((vertical_edges + horizontal_edges) / (h + w))
        return entry
    
    
    
    
    def _artifact_score(self, img: Image.Image, entry: Dict[str, Any]) -> Dict[str, Any]:
        gray = np.asarray(img.convert("L"), dtype=np.float32)
        h, w = gray.shape

        # Compute valid 8px boundaries
        vertical_positions = np.arange(8, w, 8)
        horizontal_positions = np.arange(8, h, 8)

        if len(vertical_positions) > 0:
            vertical = np.abs(
                gray[:, vertical_positions] - gray[:, vertical_positions - 1]
            ).mean()
        else:
            vertical = 0.0

        if len(horizontal_positions) > 0:
            horizontal = np.abs(
                gray[horizontal_positions, :] - gray[horizontal_positions - 1, :]
            ).mean()
        else:
            horizontal = 0.0

        entry["artifact_score"] = float((vertical + horizontal) / 2.0)
        return entry

            
        
        
    
    
    

    def analyze_image_batch(
        self,
        image_batch: List[Image.Image],
        data_batch: List[Tuple[str, Dict[str, Any], str, str]],
    ) -> List[Tuple[str, Dict[str, Any], str, str]]:
        total_images = len(image_batch)
        total_data = len(data_batch)
        if total_images != total_data:
            raise IndexError(
                f"image batch ({total_images}) doesn't match data batch ({total_data})"
            )
        for i in range(total_images):
            current_image = image_batch[i]
            current_data = data_batch[i]
            current_entry = current_data[1]
            current_entry = self._artifact_score(current_image, current_entry)
            current_entry = self._colorfulness(current_image, current_entry)
            current_entry = self._contrast(current_image, current_entry)
            current_entry = self._image_size(current_image, current_entry)
            current_entry = self._noise_score(current_image, current_entry)
            current_entry = self._sharpness(current_image, current_entry)

            processed_data = (
                current_data[0],
                current_entry,
                current_data[2],
                current_data[3],
            )
            data_batch[i] = processed_data
        return data_batch


 

    def analyze_images_from_paths(
        self,
        batch_size: int = 4,
        max_workers: Optional[int] = None,
    ) -> List[Tuple[str, Dict[str, Any], str, str]]:
        batch_size=32
        max_workers=64
        total: int = len(self.path_list)
        num_batches: int = (total + batch_size - 1) // batch_size

        if total == 0:
            self.processed_data = []
            return self.processed_data

        # Default: CPU-bound workload → match CPU count
        if max_workers is None:
            max_workers = os.cpu_count() or 1

        # Build batch argument list explicitly
        batches = [
            (
                self.prepare_image_batch,
                self.analyze_image_batch,
                self.path_list[i : i + batch_size],
                self.raw_data[i : i + batch_size],
            )
            for i in range(0, total, batch_size)
        ]

        results_nested: List[List[Tuple[str, Dict[str, Any], str, str]]] = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_single_batch, *batch_args)
                for batch_args in batches
            ]

            try:
                with tqdm(total= total, desc="Analyzed", unit=" files") as pbar:
                    for future in as_completed(futures):
                        result = future.result()  # raises immediately if a batch fails
                        results_nested.append(result)
                        pbar.update(len(result))

            except Exception:
                for f in futures:
                    f.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
                raise

        # Flatten
        self.processed_data = [
            item for batch in results_nested for item in batch
        ]

        print(f"total analyzed images: {len(self.processed_data)}")

        return self.processed_data