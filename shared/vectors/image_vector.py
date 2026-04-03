from PIL import Image
from typing import Any
from tqdm import tqdm
from collections import defaultdict, OrderedDict
import torch
from torchvision import transforms
import numpy as np

from ..loaders.model_loader import model_loader
from ..config import config
from .helpers import l2_normalize_batch
from ..io import load_json, atomic_write_json
from ..paths import vectors_size_file


class ImageVector:
    def __init__(self, name: str) -> None:
        self.name = name
        self.image_list: list[Image.Image]=[]
        self.path_list: list[str] = []
        self.vector_list: list[list[float]] = []
        self.model: Any = None
        self.vector_length: int = 0
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
        transformed_images: list[torch.Tensor] = [
            self._transform(img) for img in current_batch
        ]
        # sprint(f"transformed images[0] type: {type(transformed_images[0])}")
        device = next(model.parameters()).device

        batch_tensor = torch.stack(transformed_images, dim=0).to(
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

    def get_batch_size(self, width: int, height: int, rebuild: bool = False) -> int:
        """
        Estimate realistic peak GPU memory in bytes for a single image.
        Measures the memory used by activations + intermediate buffers,
        excluding model weights.

        Uses the device specified in your vision model config.
        """
        # only use dedicated memory
        torch.cuda.set_per_process_memory_fraction(0.99, 0)
        if self.model is None:
            self.model, self.vector_length, total_memory = model_loader.load_vision_model()

        width_str = str(width)
        height_str = str(height)
        vector_width: dict[str, int] = {}
        if self.vector_sizes:
            if width_str in self.vector_sizes:
                vector_width = self.vector_sizes[width_str]
                if height_str in vector_width and not rebuild:
                    return vector_width[height_str]

        # Get device from config
        self.prepare_config = config["prepare"]
        vision_model_config = self.prepare_config["vision_model"]
        device: str = vision_model_config["device"]

        # --------------------------
        # Step 1: Measure model-only memory
        # --------------------------
        total_memory = int(torch.cuda.get_device_properties(device).total_memory)
        torch.backends.cudnn.benchmark = True

        channels = 3

        max_size = 200
        min_size = 2
        last_size = 0

        # if vector_width:
        #     # Get all recorded heights for this width as integers
        #     existing_heights = sorted([int(h) for h in vector_width.keys()])

        #     # upper values (larger heights) => smaller batch size (min_size)
        #     # upper_heights = [h for h in existing_heights if h > height]
        #     # if upper_heights:
        #     #     min_size = vector_width[str(min(upper_heights))]

        #     # lower values (smaller heights) => larger batch size (max_size)
        #     lower_heights = [h for h in existing_heights if h < height]
        #     if lower_heights:
        #         max_size = vector_width[str(max(lower_heights))]

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
                f"for size ({width},{height}) trying batch size {current_size} (min: {min_size}, max: {max_size})....",
                flush=True,
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

        batch_size = max(int(max_size), 1)
        print(f"setting final size: {batch_size}")

        # Set value for WxH
        vector_width[height_str] = batch_size
        vector_width = dict(
            sorted(vector_width.items(), reverse=True, key=lambda x: int(x[0]))
        )
        self.vector_sizes[width_str] = vector_width

        # Set value for HxW
        if height_str not in self.vector_sizes:
            self.vector_sizes[height_str] = {}
        vector_height = self.vector_sizes[height_str]
        vector_height[width_str] = batch_size
        vector_height = dict(
            sorted(vector_height.items(), reverse=True, key=lambda x: int(x[0]))
        )
        self.vector_sizes[height_str] = vector_height

        # Combine and sort full list
        self.vector_sizes = dict(
            sorted(self.vector_sizes.items(), reverse=True, key=lambda x: int(x[0]))
        )
        atomic_write_json(vectors_size_file, self.vector_sizes, indent=4)
        return batch_size

    def create_vector_list(self,
        memory_usage: float = 0.85,rebuild:bool=False) -> list[list[float]]| None:
        
        self.model, self.vector_length, _ = model_loader.load_vision_model()
        batch_size = self.get_batch_size(
            self.image_list[0].size[0], self.image_list[0].size[1], rebuild
        )
        batch_size = int(batch_size * memory_usage)
        print(f"batch size: {batch_size} for image size: {self.image_list[0].size}")
        for i in range(0, len(self.image_list), batch_size):
            current_batch = self.image_list[i : i + batch_size]
            try:
                current_processed_images = self .create_image_vector_batch(
                    current_batch
                )
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
    ) -> list[list[float]]| tuple[int,int]:
        """
        Exact-size bucketing with controlled RAM and VRAM usage.

        - Images are NOT stored globally.
        - Only paths + indices are stored.
        - Images are loaded per batch only.
        - Order of vectors is preserved.
        """

        self.model, self.vector_length, total_memory = model_loader.load_vision_model()

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
            with Image.open(img_path) as img:
                size = img.size  # (width, height)
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
                    if rebuild_width == width and rebuild_height == height:
                        print(f"rebuild triggered by size: {size}")
                        rebuild = True
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
                            batch_vecs = self.create_image_vector_batch(current_batch)
                        except Exception:
                            print("error while encoding images, retrying...")
                            del current_batch
                            del batch_vecs
                            del vectors
                            pbar.close()
                            bucket_pbar.close()
                            torch.cuda.empty_cache()
                            return (width,height)

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
