import torch
from typing import Dict, Any, List, Tuple
import numpy as np

from ...shared.helpers import export_image_batch
from ...shared.vectors.vectors import VectorList
from ...shared.image_analysis import ImageAnalysis
from ...shared.vectors.image_vector import ImageVector
from ...shared.training.data_transformer import data_transformer
from ...shared.loaders.training_loader import training_loader


class AestheticScoreNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "image": ("IMAGE",),  # [B, H, W, C],
                "threshold": (
                    "FLOAT",
                    {"default": 2.5, "min": 0.1, "max": 5.0, "step": 0.1},
                ),
                "positive": ("STRING", {"multiline": True}),
                "negative": ("STRING", {"multiline": True}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 1000}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0}),
                "sampler": ("STRING", {"default": "euler"}),
                "scheduler": ("STRING", {"default": "normal"}),
                "model_name": ("STRING", {"default": "unknown"}),
                "lora_name": ("STRING", {"default": "unknown"}),
                "lora_strength": ("FLOAT", {"default": 0.0}),
                "min_images": ("INT", {"default": 1, "min": 0, "max": 100}),
                "max_images": ("INT", {"default": 10, "min": 0, "max": 100}),
            }
        }

    RETURN_TYPES = (
        "IMAGE",
        "IMAGE",
        "BOOLEAN",
        "LIST",
    )
    RETURN_NAMES = ("images", "discarded images", "Available", "score")
    FUNCTION = "calculate_score"
    CATEGORY = "Scoring"

    def calculate_score(
        self,
        image: torch.Tensor,
        threshold: float,
        positive: str,
        negative: str,
        steps: int,
        cfg: float,
        sampler: str,
        scheduler: str,
        model_name: str,
        lora_name: str,
        lora_strength: float,
        min_images: int = 1,
        max_images: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor, bool, List[float]]:

        batch_size = 10

        # # Normalize incoming image(s) to a list of PIL.Images.
        # try:
        #     images = prepare_image_batch(image)
        # except Exception as e:
        #     raise TypeError(
        #         f"'image' could not be processed into a batch of images: {e}"
        #     )
        if len(image) < 1:
            raise ValueError("'image' must contain at least one image.")
        if not positive.strip():
            raise ValueError("The 'positive' prompt must be a non-empty string.")
        if not negative.strip():
            raise ValueError("The 'negative' prompt must be a non-empty string.")
        if not (1 <= steps <= 1000):
            raise ValueError("'steps' must be an integer between 1 and 1000.")
        if not (0.0 <= float(cfg) <= 100.0):
            raise ValueError("'cfg' must be a float between 0.0 and 100.0.")
        if not sampler.strip():
            raise ValueError("'sampler' must be a non-empty string.")
        if not scheduler.strip():
            raise ValueError("'scheduler' must be a non-empty string.")
        if not model_name.strip():
            raise ValueError("'model_name' must be a non-empty string.")
        if not (float(lora_strength) >= 0.0):
            raise ValueError("'lora_strength' must be a non-negative float.")

        entry: Dict[str, Any] = {
            "steps": steps,
            "cfg": cfg,
            "sampler": sampler,
            "scheduler": scheduler,
            "model_name": model_name,
            "lora_name": lora_name,
            "lora_strength": lora_strength,
            "positive_prompt": positive,
            "negative_prompt": negative,
            # "score":-1
        }
        image_analysis = ImageAnalysis([])
        images_list = image_analysis.prepare_image_batch(image)
        processed_data: List[Tuple[str, Dict[str, Any], str, str]] = []
        for i in range(0, len(images_list), batch_size):
            current_batch = images_list[i : i + batch_size]
            data: Tuple[str, Dict[str, Any], str, str] = ("", entry, "", "")
            data_batch: List[Tuple[str, Dict[str, Any], str, str]] = [data] * len(
                current_batch
            )
            data_batch = image_analysis.analyze_image_batch(current_batch, data_batch)
            processed_data.extend(data_batch)

        vector_list = VectorList(
            processed_data, [], [], [], add_new=False, merge_lists=False, read_only=True
        )
        vector_list.create_vectors()
        image_vector: ImageVector = vector_list.sorted_vectors["image"]["vector"]
        batch_size = image_vector.get_batch_size(
            images_list[0].size[0], images_list[0].size[1]
        )
        processed_images: List[List[float]] = []
        for i in range(0, len(images_list), batch_size):
            current_batch = images_list[i : i + batch_size]
            current_processed_images = image_vector.create_image_vector_batch(
                current_batch
            )
            processed_images.extend(current_processed_images)

        image_vector.vector_list = processed_images
        vector_list.sorted_vectors["image"]["vector"] = image_vector

        final_vectors = vector_list.join_vectors()
        matrix = np.array(final_vectors, dtype=np.float32)

        filtered_vectors = data_transformer.apply_feature_filter(matrix)
        model = training_loader.load_training_model()
        all_scores = model.predict(filtered_vectors)
        print(f"all_scores: {all_scores}")

        # Now we have `images_list` and `all_scores` matching by index
        n_images = len(images_list)
        if len(all_scores) != n_images:
            raise RuntimeError(
                "Scoring produced a different number of scores than images"
            )

        # Sort images by score (descending)
        sorted_indices = sorted(
            range(n_images), key=lambda i: all_scores[i], reverse=True
        )

        # Select images above threshold first
        selected: List[int] = [i for i in sorted_indices if all_scores[i] >= threshold]

        # If not enough images, fill using top-scoring remaining images
        if len(selected) < int(min_images):
            for i in sorted_indices:
                if i not in selected:
                    selected.append(i)
                if len(selected) >= int(min_images):
                    break

        # Apply max_images limit (0 means unlimited)
        if int(max_images) > 0 and len(selected) > int(max_images):
            selected = selected[: int(max_images)]

        # Build selected and discarded lists in descending score order
        selected_sorted = sorted(selected, key=lambda i: all_scores[i], reverse=True)
        discarded = [i for i in sorted_indices if i not in selected_sorted]

        selected_images = [images_list[i] for i in selected_sorted]
        discarded_images = [images_list[i] for i in discarded]

        return (
            export_image_batch(selected_images),
            export_image_batch(discarded_images),
            len(selected_images) > 0,
            all_scores,
        )
