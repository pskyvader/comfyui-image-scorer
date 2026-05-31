from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from typing import Any
from tqdm import tqdm
from collections import defaultdict, OrderedDict
import torch
from torch import Tensor
from torchvision import transforms
import torch as th
import numpy as np
import math
import time

from ..loaders.model_loader import model_loader
from ..config import config
from .helpers import l2_normalize_batch
from ..io import load_json, atomic_write_json
from ..paths import vectors_size_file
from .batch_sizer import BatchSizer

import logging

logger = logging.getLogger(__name__)


class ImageConverter:
    @staticmethod
    def to_pil(arr: Any) -> list[Image.Image]:
        arr_np = np.asarray(arr)
        if arr_np.ndim == 4:
            result: list[Image.Image] = []
            for index in range(arr_np.shape[0]):
                result.extend(ImageConverter.to_pil(arr_np[index]))
            return result
        if arr_np.ndim == 3:
            if arr_np.shape[0] in (1, 3, 4):
                arr_np = np.transpose(arr_np, (1, 2, 0))
            if arr_np.shape[2] == 1:
                arr_np = np.repeat(arr_np, 3, axis=2)
            if arr_np.shape[2] == 4:
                arr_np = arr_np[:, :, :3]
            if np.issubdtype(arr_np.dtype, np.floating):
                arr_np = np.clip(arr_np, 0.0, 1.0)
                arr_np = (arr_np * 255.0).round().astype(np.uint8)
            else:
                arr_np = np.clip(arr_np, 0, 255).astype(np.uint8)
            result = [Image.fromarray(arr_np).convert("RGB")]
            return result
        if arr_np.ndim == 2:
            if np.issubdtype(arr_np.dtype, np.floating):
                arr_np = np.clip(arr_np, 0.0, 1.0)
                arr_np = (arr_np * 255.0).round().astype(np.uint8)
            arr_np = np.stack([arr_np] * 3, axis=-1)
            result = [Image.fromarray(arr_np).convert("RGB")]
            return result
        raise ValueError(f"Unsupported ndarray shape: {arr_np.shape}")

    @staticmethod
    def from_path(image_path: str) -> list[Image.Image]:
        try:
            with Image.open(image_path) as image:
                result = [image.convert("RGB")]
        except Exception:
            try:
                with Image.open(image_path) as temp_img:
                    size = temp_img.size
            except Exception:
                size = (512, 512)
            result = [Image.new("RGB", size, (0, 0, 0))]
        return result

    @staticmethod
    def prepare(image: Any) -> list[Image.Image]:
        if isinstance(image, str):
            result = ImageConverter.from_path(image)
        elif isinstance(image, Image.Image):
            result = [image.convert("RGB")]
        elif isinstance(image, torch.Tensor):
            result = ImageConverter.to_pil(image.detach().cpu().numpy())
        elif isinstance(image, np.ndarray):
            result = ImageConverter.to_pil(image)
        elif isinstance(image, (list, tuple)):
            result = []
            for item in image:
                result.extend(ImageConverter.prepare(item))
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        return result

    @staticmethod
    def get_size_from_path(img_path: str) -> tuple[int, int]:
        try:
            with Image.open(img_path) as image:
                result = image.size
        except Exception:
            result = (512, 512)
        return result


class VectorEncoder:
    _transform: transforms.Compose | None = None

    @classmethod
    def get_transform(cls) -> transforms.Compose:
        if cls._transform is None:
            cls._transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225),
                    ),
                ]
            )
        return cls._transform

    @staticmethod
    def encode(
        images: list[Image.Image],
        model: torch.nn.Module,
        vector_length: int,
        transform: transforms.Compose,
    ) -> list[list[float]]:
        transformed_images = [transform(img) for img in images]
        device = next(model.parameters()).device
        batch_tensor = torch.stack(transformed_images, dim=0).to(
            device, non_blocking=True
        )
        with torch.no_grad():
            outputs = model(batch_tensor)
        _, output_length = outputs.shape
        if output_length != vector_length:
            raise RuntimeError(f"Unexpected vector length {output_length}")
        processed = outputs.detach().cpu().float().numpy()
        normalized = l2_normalize_batch(processed)
        result = normalized.tolist()
        return result

    @staticmethod
    def run_test(
        model: torch.nn.Module, batch_tensor: torch.Tensor, device: str
    ) -> None:
        with torch.inference_mode():
            model.eval()
            model(batch_tensor)
            torch.cuda.synchronize(device)


class PathProcessor:
    @staticmethod
    def build_buckets(
        path_list: list[str],
    ) -> dict[tuple[int, int], list[tuple[int, str]]]:
        buckets: dict[tuple[int, int], list[tuple[int, str]]] = defaultdict(list)
        for index, path in enumerate(path_list):
            size = ImageConverter.get_size_from_path(path)
            buckets[size].append((index, path))
        return buckets

    @staticmethod
    def sort_buckets(
        buckets: dict[tuple[int, int], list[tuple[int, str]]],
    ) -> dict[tuple[int, int], list[tuple[int, str]]]:
        result = dict(
            sorted(
                buckets.items(),
                key=lambda item: item[0][0] * item[0][1],
                reverse=True,
            )
        )
        return result

    @staticmethod
    def process_bucket(
        items: list[tuple[int, str]],
        size: tuple[int, int],
        model: torch.nn.Module,
        vector_length: int,
        transform: transforms.Compose,
        batch_sizer: Any,
        memory_usage: float,
        vectors: list[list[float]],
        pbar: tqdm,
        rebuild: bool = False,
    ) -> tuple[int, int] | None:
        width, height = size
        batch_size = max(int(batch_sizer.get(width, height, rebuild) * memory_usage), 1)

        for start in range(0, len(items), batch_size):
            batch_slice = items[start : start + batch_size]
            batch_indices = [index for index, _ in batch_slice]
            batch_paths = [path for _, path in batch_slice]

            try:
                current_batch = ImageConverter.prepare(batch_paths)
                batch_vectors = VectorEncoder.encode(
                    current_batch,
                    model,
                    vector_length,
                    transform,
                )
            except Exception:
                torch.cuda.empty_cache()
                retry_batch_size = max(
                    int(batch_sizer.get(width, height, True) * memory_usage), 1
                )
                try:
                    current_batch = ImageConverter.prepare(batch_paths)
                    batch_vectors = VectorEncoder.encode(
                        current_batch,
                        model,
                        vector_length,
                        transform,
                    )
                except Exception:
                    return size
                batch_size = retry_batch_size

            for index, vector in zip(batch_indices, batch_vectors):
                vectors[index] = vector
            pbar.update(len(batch_slice))

        result = None
        return result


class ImageVector:
    def __init__(self, name: str) -> None:
        self.name = name
        self.image_list: list[Image.Image] = []
        self.path_list: list[str] = []
        self.vector_list: list[list[float]] = []
        self.model: Any = None
        self.vector_length: int = 0
        self.batch_sizer = BatchSizer()
        self._transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

        try:
            self.vector_sizes, _ = load_json(vectors_size_file, expect=dict, default={})
        except:
            self.vector_sizes = {}

        # only use dedicated memory
        torch.cuda.set_per_process_memory_fraction(0.99, 0)

    def array_to_pil(self, arr: Any) -> list[Image.Image]:
        arr = np.asarray(arr)
        if arr.ndim == 4:
            # Batch of images
            out: list[Image.Image] = []
            for i in range(arr.shape[0]):
                sub = self.array_to_pil(arr[i])
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

    def prepare_image_batch(self, image: Any) -> list[Image.Image]:
        """
        Convert various image inputs into a list of RGB PIL Images.
        Accepts torch.Tensor, numpy arrays, PIL.Image, or list/tuple of these.
        Returns a list of `PIL.Image.Image` (RGB).
        """

        # Start handling different input types
        if isinstance(image, str):
            try:
                img = Image.open(image).convert("RGB")
                return [img]
            except Exception as e:
                logger.warning(f"Warning: (retrying) Failed to load image {image}: {e}")
                try:
                    with Image.open(image) as temp_img:
                        size = temp_img.size
                except Exception as e:
                    logger.error(f"Failed to load image {image}: {e}")
                    raise
                return [Image.new("RGB", size, (0, 0, 0))]
        if isinstance(image, Image.Image):
            return [image.convert("RGB")]
        if isinstance(image, torch.Tensor):
            res = self.array_to_pil(image.detach().cpu().numpy())
            return res
        if isinstance(image, np.ndarray):
            res = self.array_to_pil(image)
            return res
        if isinstance(image, (list, tuple)):
            out: list[Image.Image] = []
            for it in image:
                out.extend(self.prepare_image_batch(it))
            return out
        raise TypeError(f"Unsupported image type: {type(image)}")

    def create_image_vector_batch(
        self, current_batch: list[Image.Image]
    ) -> list[list[float]]:
        """
        Encodes a batch of images into vectors. Assumes all images in the batch
        have the same size.
        """

        # All images have the same size (width, height)
        # batch_size = len(current_batch)
        # img_size = current_batch[0].size  # (width, height)
        # batch_shape = (batch_size, img_size[0], img_size[1])

        model = self.model
        transformed_images = [self._transform(img) for img in current_batch]
        device = next(model.parameters()).device

        batch_tensor = torch.stack(transformed_images, dim=0).to(  # type: ignore[arg-type]
            device, non_blocking=True
        )

        with torch.no_grad():
            outputs = model(batch_tensor)

        # Validate output shape
        _, vector_length = outputs.shape
        if vector_length != self.vector_length:
            raise RuntimeError(f"Unexpected vector length {vector_length}")

        processed = outputs.detach().cpu().float().numpy()
        normalized = l2_normalize_batch(processed)
        return normalized.tolist()

    def test_batch(self, batch_tensor: torch.Tensor, device: str) -> None:
        with torch.inference_mode():
            # Ensure model is in eval mode so dropout/batchnorm don't waste memory
            self.model.eval()

            # Full pass is the only way to catch the "21 vs 85" discrepancy
            self.model(batch_tensor)

            # The "Force-Fail": Without this, Python won't know the GPU crashed
            # until it's too late to catch it in this loop.
            torch.cuda.synchronize(device)

    def get_batch_size(self, width: int, height: int, rebuild: bool = False) -> int:
        result = self.batch_sizer.get(width, height, rebuild)
        return result

    def create_vector_list(
        self, memory_usage: float = 0.85, rebuild: bool = False
    ) -> list[list[float]] | None:
        if not self.image_list:
            result: list[list[float]] = []
            return result
        self.model, self.vector_length, _ = model_loader.load_vision_model()
        batch_size: int = self.get_batch_size(
            self.image_list[0].size[0], self.image_list[0].size[1], rebuild
        )
        batch_size = int(batch_size * memory_usage)
        print(f"batch size: {batch_size} for image size: {self.image_list[0].size}")
        for i in range(0, len(self.image_list), batch_size):
            current_batch = self.image_list[i : i + batch_size]
            try:
                current_processed_images = self.create_image_vector_batch(current_batch)
            except Exception:
                print("error while encoding images, retrying...")
                return None

            self.vector_list.extend(current_processed_images)
        return self.vector_list

    def create_vector_list_from_paths(
        self,
        memory_usage: float = 0.85,
        rebuild_width: int = -1,
        rebuild_height: int = -1,
    ) -> list[list[float]] | tuple[int, int]:
        """
        Exact-size bucketing with controlled RAM and VRAM usage.

        - Images are NOT stored globally.
        - Only paths + indices are stored.
        - Images are loaded per batch only.
        - Order of vectors is preserved.
        """

        self.model, self.vector_length, total_memory = model_loader.load_vision_model()

        if rebuild_width > 0 and rebuild_height > 0:
            print(f"rebuild requested for size ({rebuild_width},{rebuild_height})")
            self.get_batch_size(rebuild_width, rebuild_height, rebuild=True)

        # self.model = model
        # self.vector_length = vector_length

        total = len(self.path_list)
        vectors: list[list[float]] = [[]] * total  # preserve original order
        if len(self.path_list) == 0:
            print("empty image list")
            return self.vector_list

        # --------------------------------------------------
        # PASS 1: Build buckets using only metadata
        # --------------------------------------------------
        buckets: dict[tuple[int, int], list[tuple[int, str]]] = defaultdict(list)

        for idx, img_path in enumerate(self.path_list):
            try:
                with Image.open(img_path) as img:
                    size = img.size  # (width, height)
            except Exception as e:
                print(f"Warning: Failed to get size for {img_path}: {e}")
                size = (512, 512)
            buckets[size].append((idx, img_path))

        sorted_buckets = OrderedDict(
            sorted(buckets.items(), key=lambda kv: kv[0][0] * kv[0][1], reverse=True)
        )

        bucket_list = list(sorted_buckets.items())
        total_buckets = len(bucket_list)

        print(f"\nTotal buckets: {total_buckets}")
        print(f"\nTotal memory: {total_memory}")
        print(f"\nTotal: {total}")

        # --------------------------------------------------
        # PASS 2: Process each bucket
        # --------------------------------------------------
        with tqdm(
            total=total_buckets,
            desc="Buckets",
            unit="bucket",
            position=0,
            leave=True,
        ) as bucket_pbar:
            with tqdm(
                total=total,
                desc="Encoded",
                unit=self.name,
                position=1,
            ) as pbar:
                max_batch_size = 0
                for bucket_idx, (size, items) in enumerate(bucket_list, start=1):
                    num_items = len(items)
                    # if num_items > max_batch_size:
                    width, height = size
                    rebuild = False
                    # if rebuild_width == width and rebuild_height == height:
                    #     print(f"rebuild triggered by size: {size}")
                    #     print(
                    #         f"rebuild width: {rebuild_width}, rebuild height: {rebuild_height}"
                    #     )
                    #     rebuild = True
                    max_batch_size = self.get_batch_size(width, height, rebuild)
                    max_batch_size = int(
                        max_batch_size * memory_usage
                    )  # max memory percentage allowed to use

                    if num_items > max_batch_size * 2:
                        torch.backends.cudnn.benchmark = True
                    else:
                        torch.backends.cudnn.benchmark = False

                    bucket_pbar.set_description(
                        f"Bucket {bucket_idx} | Size: {size} | Items: {num_items} | Batch: {max_batch_size}"
                    )

                    # Process bucket in sub-batches
                    for i in range(0, num_items, max_batch_size):

                        batch_slice = items[i : i + max_batch_size]

                        # Separate indices and paths
                        batch_indices = [idx for idx, _ in batch_slice]
                        batch_paths = [path for _, path in batch_slice]

                        # Load & preprocess batch in one call
                        current_batch = self.prepare_image_batch(batch_paths)

                        batch_vecs = None
                        try:
                            # Encode batch
                            batch_vecs: list[list[float]] | None = (
                                self.create_image_vector_batch(current_batch)
                            )
                        except Exception:
                            torch.cuda.empty_cache()
                            pbar.close()
                            bucket_pbar.close()
                            print(
                                f"error while encoding images, retrying With size: {size}..."
                            )
                            del current_batch
                            del batch_vecs
                            del vectors
                            # time.sleep(1)
                            return (width, height)

                            # return self.create_vector_list_from_paths(
                            #     memory_usage, width, height
                            # )

                        # Store vectors in original order
                        for idx, vec in zip(batch_indices, batch_vecs):
                            vectors[idx] = vec

                        pbar.update(len(batch_slice))

                        # Explicit cleanup (important for VRAM stability)
                        del current_batch
                        del batch_vecs
                    torch.cuda.empty_cache()

                    bucket_pbar.update(1)

        self.vector_list.extend(vectors)
        return self.vector_list
