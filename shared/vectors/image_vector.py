from PIL import Image
from typing import List, Any, Dict
from tqdm import tqdm
import torch
from torchvision import transforms
import numpy as np
from ..loaders.model_loader import model_loader
from ..config import config
from typing import List
from tqdm import tqdm
from collections import defaultdict, OrderedDict

from .helpers import l2_normalize_batch
from ..io import load_json, atomic_write_json
from ..paths import vectors_size_file


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
        try:
            self.vector_sizes, _ = load_json(vectors_size_file, expect=dict, default={})
        except:
            self.vector_sizes = {}

        # only use dedicated memory
        torch.cuda.set_per_process_memory_fraction(0.99, 0)

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

    def create_image_vector_batch(
        self, current_batch: List[Image.Image]
    ) -> List[List[float]]:
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

        batch_tensor = torch.stack([transform(img) for img in current_batch], dim=0).to(
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

    def get_batch_size(self, width: int, height: int, max_memory: float = 0.8) -> int:
        """
        Estimate realistic peak GPU memory in bytes for a single image.
        Measures the memory used by activations + intermediate buffers,
        excluding model weights.

        Uses the device specified in your vision model config.
        """
        width_str = str(width)
        height_str = str(height)
        vector_width: Dict[str, int] = {}
        if self.vector_sizes:
            if width_str in self.vector_sizes:
                vector_width = self.vector_sizes[width_str]
                if height_str in vector_width:
                    return vector_width[height_str]

        # Get device from config
        self.prepare_config = config["prepare"]
        vision_model_config = self.prepare_config["vision_model"]
        device: str = vision_model_config["device"]

        # --------------------------
        # Step 1: Measure model-only memory
        # --------------------------
        props = torch.cuda.get_device_properties(device)
        total_memory = int(props.total_memory)
        torch.backends.cudnn.benchmark = True

        channels = 3

        max_size = 100
        min_size = 1
        last_size = 0
        while not max_size == min_size:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            initial_mem = int(torch.cuda.max_memory_allocated(device))
            current_size = int((max_size + min_size) / 2)
            if current_size == last_size:
                max_size = min_size = current_size
                print(f"repeated size, setting final size: {current_size} ")
                break
            last_size = current_size
            print("----")
            print(
                f"for size ({width},{height}) trying batch size {current_size} (min: {min_size}, max: {max_size})...."
            )
            batch_tensor = torch.zeros(
                (current_size, channels, height, width), dtype=torch.float32
            ).to(device)

            # --------------------------
            # Step 2: Measure memory with dummy image
            # --------------------------
            try:
                with torch.no_grad():
                    _ = self.model(batch_tensor)
            except Exception as e:
                max_size = current_size
                print(e)
                print("setting new max size: ", max_size)
                continue

            final_mem = int(torch.cuda.max_memory_allocated(device))

            image_vram = final_mem - initial_mem
            print(
                f"for batch size {current_size}, memory: {image_vram} ({final_mem}-{initial_mem}) (total memory:{total_memory})"
            )
            if image_vram > total_memory:
                max_size = current_size
                print(
                    f"max dedicated memory exceeded,setting new max batch size: {max_size}"
                )
            else:
                min_size = current_size
                print("setting new min size: ", min_size)

        batch_size = max(int(max_size * max_memory), 1)
        print(
            f"setting final size: {batch_size} (max memory allowed: {max_memory*100}% )"
        )
        vector_width[height_str] = batch_size
        vector_width = dict(
            sorted(vector_width.items(), reverse=True, key=lambda x: int(x[0]))
        )
        self.vector_sizes[width_str] = vector_width
        self.vector_sizes = dict(
            sorted(self.vector_sizes.items(), reverse=True, key=lambda x: int(x[0]))
        )
        atomic_write_json(vectors_size_file, self.vector_sizes, indent=4)
        return batch_size

    def create_vector_list_from_paths(self) -> list[list[float]]:
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
                # if num_items > max_batch_size:
                width, height = size
                max_batch_size = self.get_batch_size(width, height, 0.9)

                if num_items > max_batch_size * 2:
                    torch.backends.cudnn.benchmark = True
                else:
                    torch.backends.cudnn.benchmark = False

                print(
                    f"Processing bucket {bucket_idx}/{total_buckets} | "
                    f"Image size: {size} | "
                    f"Elements in bucket: {num_items} | "
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
                torch.cuda.empty_cache()

        self.vector_list.extend(vectors)
        return self.vector_list
