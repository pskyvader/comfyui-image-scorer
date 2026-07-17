from PIL.Image import Image
import torch
from typing import Any
import numpy as np
import warnings
from ...shared.config import config
from ...shared.helpers import export_image_batch
from ...shared.vectors.vectors import VectorList
from ...shared.analysis.image_analysis import ImageAnalysis
from ...shared.vectors.image_vector import ImageVector
from ...shared.training.data_transformer import data_transformer
from ...shared.loaders.training_loader import training_loader
from ...shared.training.calibration import (
    apply_score_calibration,
    extract_score_calibration,
)


class AestheticScoreNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, Any]]:
        return {
            "required": {
                "image": ("IMAGE",),  # [B, H, W, C],
                "threshold": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05},
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

    def _predict_scores(
        self, model: Any, filtered_vectors: list[np.ndarray]
    ) -> np.ndarray:
        features = np.asarray(filtered_vectors, dtype=np.float32)
        objective = None
        if hasattr(model, "get_params"):
            try:
                objective = model.get_params().get("objective")
            except Exception:
                objective = None

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message=".*X does not have valid feature names.*",
            )
            if objective == "binary" and hasattr(model, "predict_proba"):
                probabilities = np.asarray(
                    model.predict_proba(features), dtype=np.float32
                )
                if probabilities.ndim == 2 and probabilities.shape[1] >= 2:
                    scores = probabilities[:, 1]
                else:
                    scores = np.asarray(model.predict(features), dtype=np.float32).reshape(
                        -1
                    )
            elif objective == "multiclass" and hasattr(model, "predict_proba"):
                probabilities = np.asarray(
                    model.predict_proba(features), dtype=np.float32
                )
                if probabilities.ndim == 2 and probabilities.shape[1] > 0:
                    class_values = np.asarray(
                        getattr(
                            model,
                            "classes_",
                            np.arange(probabilities.shape[1], dtype=np.float32),
                        ),
                        dtype=np.float32,
                    )
                    if class_values.shape[0] == probabilities.shape[1]:
                        expected = np.sum(
                            probabilities * class_values[np.newaxis, :], axis=1
                        )
                        class_min = float(np.min(class_values))
                        class_max = float(np.max(class_values))
                        if class_max > class_min:
                            scores = (expected - class_min) / (
                                class_max - class_min
                            )
                        else:
                            scores = expected
                    else:
                        scores = np.max(probabilities, axis=1)
                else:
                    scores = np.asarray(model.predict(features), dtype=np.float32).reshape(
                        -1
                    )
            else:
                scores = np.asarray(model.predict(features), dtype=np.float32).reshape(
                    -1
                )
                if objective == "lambdarank":
                    diagnostics = training_loader.load_training_model_diagnostics()
                    calibration = extract_score_calibration(diagnostics)
                    if calibration is not None:
                        scores = apply_score_calibration(scores, calibration)
                    elif scores.size > 1:
                        low = float(np.min(scores))
                        high = float(np.max(scores))
                        if high > low:
                            scores = ((scores - low) / (high - low)).astype(np.float32)
                        else:
                            scores = np.full(scores.shape, 0.5, dtype=np.float32)

        return np.asarray(scores, dtype=np.float32).reshape(-1)

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
        memory_usage: float = 0.85,
    ) -> tuple[torch.Tensor, torch.Tensor, bool, list[float]]:

        batch_size = 10

        assert len(image) >= 1, "'image' must contain at least one image."
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
        # if not (float(lora_strength) >= 0.0):
        #     raise ValueError("'lora_strength' must be a non-negative float.")

        entry: dict[str, Any] = {
            "steps": steps,
            "cfg": cfg,
            "sampler": sampler,
            "scheduler": scheduler,
            "model_name": model_name,
            "lora_name": lora_name,
            "lora_strength": lora_strength,
            "positive_prompt": positive,
            "negative_prompt": negative,
        }
        image_analysis = ImageAnalysis([])
        images_list: list[Image] = image_analysis.prepare_image_batch(image)
        data_list = [("", entry, "", str(i)) for i, _img in enumerate(images_list)]

        processed_data: list[tuple[str, dict[str, Any], str, str]] = []
        for i in range(0, len(images_list), batch_size):
            image_batch = images_list[i : i + batch_size]
            data_batch = data_list[i : i + batch_size]
            data_batch = image_analysis.analyze_image_batch(image_batch, data_batch)
            processed_data.extend(data_batch)

        vector_list = VectorList(
            processed_data,
            read_only=True,
        )
        vector_list.create_vectors()

        vector_config = config["vector"]["vectors"]
        for entry in (v for v in vector_config if v["type"] == "image"):
            name = entry["name"]
            model_key = entry["model_key"]
            image_vector = ImageVector(name, model_key=model_key, slot_size=entry["slot_size"])
            # image_vector.image_list = images_list
            images_dict = {str(i): image for i, image in enumerate(images_list)}
            rebuild = False
            retry = True
            while retry:
                result = image_vector.create_vector_list(images_dict, 0.85, rebuild)
                if result is None:
                    rebuild = True
                    retry = True
                else:
                    retry = False
            vector_list.sorted_vectors[name]["vector"] = image_vector

        # vector_list.filter_missing_vectors()
        # print(f"unique ids: {vector_list.unique_ids}")
        final_vectors = vector_list.join_vectors()
        # print(f"final vectors: {len(final_vectors)}")
        matrix = np.array(final_vectors, dtype=np.float32)
        # print(f"transformed vectors: {matrix.shape}")

        filtered_vectors = data_transformer.apply_feature_filter(list(matrix))
        # interaction_vectors=data_transformer.apply_interaction_features(filtered_vectors)

        model = training_loader.load_training_model()  # type: ignore[union-attr]
        # print(
        #     f"filtered vectors:{len(filtered_vectors)}, shape: {filtered_vectors[0].shape}"
        # )
        all_scores = self._predict_scores(model, filtered_vectors)
        # print(f"all_scores: {all_scores}")

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
        selected: list[int] = [i for i in sorted_indices if all_scores[i] >= threshold]

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
