from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from typing import Any
from tqdm import tqdm
from collections import defaultdict, OrderedDict
import torch
from torchvision import transforms
import numpy as np

from ..loaders.model_loader import model_loader
from .helpers import l2_normalize_batch
from ..io import load_json
from ..paths import vectors_size_file
from .batch_sizer import BatchSizer

# import logging

# logger = logging.getLogger(__name__)

from ..logger import (
    # SSELogBroadcaster,
    # SharedLogger,
    get_logger,
    # set_log_filter_hook,
)

logger = get_logger(__name__)

sizeTuple = tuple[int, int]
vectorDict = dict[str, list[float]]
imagePathTuple = tuple[str, str]
imageTuple = tuple[str, Image.Image]


class ImageVector:
    def __init__(self, name: str, model_key: str) -> None:
        self.name = name
        self.model_key = model_key
        self.image_list: dict[str, Image.Image] = {}
        self.path_list: dict[str, str] = {}
        self.vector_list: vectorDict = {}
        self.model: Any = None
        self.vector_length: int = 0
        self._transform: transforms.Compose | None = None
        self.variable_input: bool = True
        self.model_input_size: sizeTuple | None = None
        self.batch_sizer = BatchSizer(model_key=model_key)

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

    def create_image_vector_batch(self, current_batch: list[imageTuple]) -> vectorDict:
        """
        Encodes a batch of images into vectors. Assumes all images in the batch
        have the same size.
        """
        if self._transform is None:
            raise RuntimeError("Model transform not set. Cannot process batch.")
        batch_id, image_batch = zip(*current_batch)

        model = self.model
        transformed_images: list[torch.Tensor] = [
            self._transform(img) for img in image_batch
        ]
        device = next(model.parameters()).device

        batch_tensor = torch.stack(transformed_images, dim=0).to(  # type: ignore[arg-type]
            device, non_blocking=True
        )

        with torch.no_grad():
            outputs = model(batch_tensor)

        # Validate output shape
        _, vector_length = outputs.shape
        if vector_length != self.vector_length:
            raise RuntimeError(
                f"Unexpected vector length {vector_length} != {self.vector_length}"
            )

        processed = outputs.detach().cpu().float().numpy()
        normalized = l2_normalize_batch(processed)

        normalized_list = normalized.tolist()

        result: dict[str, list[float]] = dict(zip(batch_id, normalized_list))
        return result

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
        self,
        entries: dict[str, Image.Image],
        memory_usage: float = 0.85,
        rebuild: bool = False,
        model_key: str | None = None,
    ) -> vectorDict | None:
        if model_key is not None:
            self.model_key = model_key
        if not entries:
            result: vectorDict = {}
            return result
        self.model, self.vector_length, _, self._transform = (
            model_loader.load_vision_model(self.model_key)
        )
        model_info = model_loader.get_model_info(self.model_key)
        self.variable_input = model_info["variable_input"]
        self.model_input_size = sizeTuple(
            model_info["input_size"]
            if not self.variable_input and model_info["input_size"] is not None
            else list(entries.values())[0].size
        )
        bw, bh = self.model_input_size
        batch_size: int = self.get_batch_size(bw, bh, rebuild)
        batch_size = max(int(batch_size * memory_usage), 1)
        logger.debug(
            f"batch size: {batch_size} for image size: {self.model_input_size}"
        )
        for i in range(0, len(entries.values()), batch_size):
            current_batch: list[imageTuple] = list(entries.items())[i : i + batch_size]
            current_processed_images: vectorDict = self.create_image_vector_batch(
                current_batch
            )

            self.vector_list.update(current_processed_images)
        return self.vector_list

    def create_vector_list_from_paths(
        self,
        entries: dict[str, str],
        memory_usage: float = 0.85,
        rebuild_width: int = -1,
        rebuild_height: int = -1,
    ) -> vectorDict | sizeTuple:
        """
        Exact-size bucketing with controlled RAM and VRAM usage.

        - Images are NOT stored globally.
        - Only paths + indices are stored.
        - Images are loaded per batch only.
        - Order of vectors is preserved.
        """

        logger.debug(
            f"Creating vector list from paths for {self.name} ({self.model_key})..."
        )

        self.model, self.vector_length, total_memory, self._transform = (
            model_loader.load_vision_model(self.model_key)
        )
        model_info = model_loader.get_model_info(self.model_key)
        self.variable_input = model_info["variable_input"]
        self.model_input_size = sizeTuple(model_info["input_size"])

        if rebuild_width > 0 and rebuild_height > 0:
            logger.debug(
                f"rebuild requested for size ({rebuild_width},{rebuild_height})"
            )
            self.get_batch_size(rebuild_width, rebuild_height, rebuild=True)

        # self.model = model
        # self.vector_length = vector_length

        total = len(entries)
        vectors: vectorDict = {}
        if len(entries) == 0:
            logger.debug("empty image list")
            return self.vector_list

        # --------------------------------------------------
        # PASS 1: Build buckets using only metadata
        # --------------------------------------------------
        buckets: dict[sizeTuple, list[imagePathTuple]] = defaultdict(list)

        for idx, img_path in entries.items():
            with Image.open(img_path) as img:
                size: sizeTuple = img.size  # (width, height)
            buckets[size].append((idx, img_path))

        sorted_buckets = OrderedDict(
            sorted(buckets.items(), key=lambda kv: kv[0][0] * kv[0][1], reverse=True)
        )

        bucket_list: list[tuple[sizeTuple, list[imagePathTuple]]] = list(
            sorted_buckets.items()
        )

        # if variable size is false, collapse all buckets into a single one
        if not self.variable_input:
            all_items: list[imagePathTuple] = []
            for size, items in bucket_list:
                all_items.extend(items)

            bucket_list = [(self.model_input_size, all_items)]

        total_buckets = len(bucket_list)

        logger.debug(f"\nTotal buckets: {total_buckets}")
        logger.debug(f"\nTotal memory: {total_memory}")
        logger.debug(f"\nTotal: {total}")

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
                    num_items: int = len(items)
                    # if num_items > max_batch_size:
                    width, height = sizeTuple(
                        self.model_input_size
                        if (not self.variable_input and self.model_input_size)
                        else size
                    )
                    # if rebuild_width == width and rebuild_height == height:
                    #     print(f"rebuild triggered by size: {size}")
                    #     print(
                    #         f"rebuild width: {rebuild_width}, rebuild height: {rebuild_height}"
                    #     )
                    #     rebuild = True
                    max_batch_size = self.get_batch_size(width, height, False)
                    max_batch_size = int(
                        max_batch_size * memory_usage
                    )  # max memory percentage allowed to use

                    if num_items > max_batch_size * 2:
                        torch.backends.cudnn.benchmark = True
                    else:
                        torch.backends.cudnn.benchmark = False

                    bucket_pbar.set_description(
                        f"Bucket {bucket_idx} | Size: {size} | Items: {num_items} | Max Batch size: {max_batch_size}"
                    )

                    # Process bucket in sub-batches
                    for i in range(0, num_items, max_batch_size):

                        batch_slice = items[i : i + max_batch_size]
                        batch_indices, batch_paths = zip(*batch_slice)

                        # Load & preprocess batch in one call
                        current_image_batch: list[Image.Image] = (
                            self.prepare_image_batch(batch_paths)
                        )
                        current_batch: list[imageTuple] = list(
                            zip(batch_indices, current_image_batch)
                        )

                        batch_vecs: vectorDict = {}
                        try:
                            # Encode batch
                            batch_vecs = self.create_image_vector_batch(current_batch)
                        except Exception as e:
                            torch.cuda.empty_cache()
                            pbar.close()
                            bucket_pbar.close()
                            logger.warning(str(e))
                            logger.warning(
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
                        vectors.update(batch_vecs)

                        pbar.update(len(batch_slice))

                        # Explicit cleanup (important for VRAM stability)
                        del current_batch
                        del batch_vecs
                    torch.cuda.empty_cache()

                    bucket_pbar.update(1)

        self.vector_list.update(vectors)
        return self.vector_list
