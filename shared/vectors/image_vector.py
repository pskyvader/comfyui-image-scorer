from PIL import Image
from typing import List, Any
from tqdm import tqdm
import torch
from torch import nn
from torchvision import transforms
import numpy as np
from ..loaders.model_loader import model_loader
from ..config import config
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional
from tqdm import tqdm
import os
from collections import defaultdict, OrderedDict


class ImageVector:
    def __init__(self, name: str) -> None:
        self.name = name
        # self.image_list: List[Image.Image]
        self.path_list: List[str] = []
        self.vector_list: List[List[float]] = []
        self._transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

        # Dictionary to track seen batch shapes
        self._seen_shapes = defaultdict(bool)

    def array_to_pil(self, arr: Any) -> List[Image.Image]:
        arr = np.asarray(arr)
        if arr.ndim == 4:
            # Batch of images
            out: List[Image.Image] = []
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
            img = Image.open(image).convert("RGB")
            return [img]
        if isinstance(image, Image.Image):
            return [image.convert("RGB")]
        if isinstance(image, torch.Tensor):
            res = self.array_to_pil(image.detach().cpu().numpy())
            return res
        if isinstance(image, np.ndarray):
            res = self.array_to_pil(image)
            return res
        if isinstance(image, (list, tuple)):
            out: List[Image.Image] = []
            for it in image:
                out.extend(self.prepare_image_batch(it))
            return out
        raise TypeError(f"Unsupported image type: {type(image)}")

    def create_image_vector_batch(self, current_batch: List[Image.Image]) -> List[List[float]]:
        """
        Encodes a batch of images into vectors. Assumes all images in the batch
        have the same size.
        """

        # All images have the same size (width, height)
        # batch_size = len(current_batch)
        # img_size = current_batch[0].size  # (width, height)
        # batch_shape = (batch_size, img_size[0], img_size[1])

        model = self.model
        transform = self._transform
        device = next(model.parameters()).device

        # # --- Selectively enable benchmark ---
        # if self._seen_shapes[batch_shape]:
        #     # This shape has been seen before → enable benchmark
        #     torch.backends.cudnn.benchmark = True
        # else:
        #     # First time seeing this shape → disable benchmark to skip profiling overhead
        #     torch.backends.cudnn.benchmark = False
        #     self._seen_shapes[batch_shape] = True

        # --- Prepare batch tensor ---
        batch_tensor = torch.stack([transform(img) for img in current_batch], dim=0).to(
            device, non_blocking=True
        )

        with torch.no_grad():
            outputs = model(batch_tensor)

        # Validate output shape
        _, vector_length = outputs.shape
        if vector_length != self.vector_length:
            raise RuntimeError(f"Unexpected vector length {vector_length}")

        return outputs.detach().cpu().float().numpy().tolist()

    # def create_image_vector_batch(
    #     self, current_batch: List[Image.Image]
    # ) -> List[List[float]]:
    #     (model, processor, vector_length) = model_loader.load_vision_model()
    #     inputs = processor(images=current_batch, return_tensors="pt")
    #     inputs = {k: v.to(model.device) for k, v in inputs.items()}
    #     with torch.no_grad():
    #         outputs = model.get_image_features(**inputs)
    #     batch_vecs = outputs.cpu().numpy().astype(float).tolist()
    #     for vec in batch_vecs:
    #         if len(vec) != vector_length:
    #             msg = f"CLIP returned unexpected vector length {len(vec)}, expected {vector_length}"
    #             raise RuntimeError(msg)
    #     return batch_vecs


    def measure_image_vram_bytes(
        self, width: int, height: int, channels: int = 3
    ) -> int:
        """
        Estimate realistic peak GPU memory in bytes for a single image.
        Measures the memory used by activations + intermediate buffers,
        excluding model weights.

        Uses the device specified in your vision model config.
        """
        # Get device from config
        self.prepare_config = config["prepare"]
        vision_model_config = self.prepare_config["vision_model"]
        device: str = vision_model_config["device"]

        # Create dummy image directly on the device
        dummy_image = torch.randn(1, channels, height, width, device=device)

        # --------------------------
        # Step 1: Measure model-only memory
        # --------------------------
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            pass  # model already loaded
        model_mem = torch.cuda.max_memory_allocated(device)

        # --------------------------
        # Step 2: Measure memory with dummy image
        # --------------------------
        # torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            _ = self.model(dummy_image)
        total_mem = torch.cuda.max_memory_allocated(device)

        # --------------------------
        # Step 3: Compute per-image VRAM
        # --------------------------
        print(f"before {model_mem}, after {total_mem}")
        image_vram = total_mem - model_mem

        # Optional safety margin (15%) for CUDA overhead / fragmentation
        image_vram = int(image_vram * 1.5)

        return image_vram

    def create_vector_list_from_paths(
        self, max_batch_size: int = 4
    ) -> list[list[float]]:
        """
        Exact-size bucketing with controlled RAM and VRAM usage.

        - Images are NOT stored globally.
        - Only paths + indices are stored.
        - Images are loaded per batch only.
        - Order of vectors is preserved.
        """

        model, vector_length, total_memory = model_loader.load_vision_model()

        self.model = model
        self.vector_length = vector_length

        total = len(self.path_list)
        vectors: List[List[float]] = [[]] * total  # preserve original order

        # --------------------------------------------------
        # PASS 1: Build buckets using only metadata
        # --------------------------------------------------
        buckets = defaultdict(list)

        for idx, img_path in enumerate(self.path_list):
            with Image.open(img_path) as img:
                size = img.size  # (width, height)
            buckets[size].append((idx, img_path))

        sorted_buckets = OrderedDict(
            sorted(buckets.items(), key=lambda kv: kv[0][0] * kv[0][1], reverse=True)
        )

        bucket_list = list(sorted_buckets.items())
        total_buckets = len(bucket_list)

        print(f"\nTotal buckets: {total_buckets}\n")
        print(f"\nTotal memory: {total_memory}\n")

        # --------------------------------------------------
        # PASS 2: Process each bucket
        # --------------------------------------------------
        with tqdm(total=total, desc="Encoded", unit=self.name) as pbar:
            max_batch_size = 0
            for bucket_idx, (size, items) in enumerate(bucket_list, start=1):

                num_items = len(items)

                if num_items > max_batch_size:
                    width, height = size
                    bytes_per_image = self.measure_image_vram_bytes(width, height)
                    # todo: transform print in secondsry tqdm
                    max_batch_size = max(
                        int((total_memory * 0.8) // bytes_per_image), max_batch_size + 1
                    )
                else:
                    bytes_per_image = -1

                if num_items > max_batch_size * 2:
                    torch.backends.cudnn.benchmark = True
                else:
                    torch.backends.cudnn.benchmark = False

                print(
                    f"Processing bucket {bucket_idx}/{total_buckets} | "
                    f"Image size: {size} | "
                    f"Elements in bucket: {num_items} | "
                    f"Bytes per image: {bytes_per_image} | "
                    f"Batch size: {max_batch_size}"
                )

                # Process bucket in sub-batches
                for i in range(0, num_items, max_batch_size):

                    batch_slice = items[i : i + max_batch_size]

                    # Separate indices and paths
                    batch_indices = [idx for idx, _ in batch_slice]
                    batch_paths = [path for _, path in batch_slice]

                    # Load & preprocess batch in one call
                    current_batch = self.prepare_image_batch(batch_paths)

                    # Encode batch
                    batch_vecs = self.create_image_vector_batch(current_batch)

                    # Store vectors in original order
                    for idx, vec in zip(batch_indices, batch_vecs):
                        vectors[idx] = vec

                    pbar.update(len(batch_slice))

                    # Explicit cleanup (important for VRAM stability)
                    del current_batch
                    del batch_vecs

                    # torch.cuda.empty_cache()

        self.vector_list.extend(vectors)
        return self.vector_list
