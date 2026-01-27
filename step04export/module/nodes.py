import os

# import sys
import numpy as np
import torch
from typing import Any, List, Dict

# try:
#     import folder_paths
# except ImportError:
#     pass  # Managed by ComfyUI
from PIL import Image
# Support being executed as a standalone module for tests (no package parent)
try:
    from .feature_assembler import (
        assemble_feature_vector,
        load_maps,
        map_categorical_value,
        apply_feature_filter,
        apply_interaction_features,
        load_prepare_config,
    )
except Exception:
    # Fallback when executed without package context (test harness exec)
    from feature_assembler import (
        assemble_feature_vector,
        load_maps,
        map_categorical_value,
        apply_feature_filter,
        apply_interaction_features,
        load_prepare_config,
    )

from scorer import ScorerModel

# Constants
SIGLIP_ID = "google/siglip-base-patch16-224"
MPNET_ID = "all-mpnet-base-v2"
IMAGE_VEC_LEN = 768


def create_image_batch(pil_images: List[Image.Image]) -> torch.Tensor:
    if not pil_images:
        return torch.zeros((1, 1, 1, 3), dtype=torch.float32)

    tensors: List[torch.Tensor] = []

    for img in pil_images:
        # 1. Ensure RGB and convert to NumPy array (Shape: H, W, C)
        # ComfyUI requires HWC format, which is the default for PIL -> NumPy
        np_img = np.array(img.convert("RGB"))

        # 2. Normalize and convert to float32
        # ComfyUI expects values between 0.0 and 1.0
        np_img = np_img.astype(np.float32) / 255.0

        # 3. Convert to Torch Tensor and add batch dimension [1, H, W, C]
        t = torch.from_numpy(np_img).unsqueeze(0)
        tensors.append(t)

    # 4. Concatenate into a single batch [B, H, W, C]
    return torch.cat(tensors, dim=0)


class AestheticScorerLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("AESTHETIC_SCORER",)
    RETURN_NAMES = ("scorer",)
    FUNCTION = "load_model"
    CATEGORY = "Scoring"

    def load_model(
        self, model_path: str = ""
    ) -> tuple[tuple[ScorerModel, Dict[str, Dict[str, int]]]]:
        # Resolve model path
        if not model_path:
            # 1. Try built-in 'models/bin' directory (Self-contained)
            internal_path = os.path.join(os.path.dirname(__file__), "models", "bin")
            model_path = internal_path

        if not os.path.isabs(model_path):
            # Resolve relative path against this file
            base = os.path.dirname(__file__)
            potential = os.path.join(base, model_path)
            if os.path.exists(potential):
                model_path = potential

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path not found: {model_path}")

        maps_path = os.path.join(os.path.dirname(__file__), "models", "maps")

        maps = load_maps(maps_path)

        # Try several candidates for prepare_config.json (node-local, repo config folder, repo root)
        node_root = os.path.dirname(__file__)
        repo_root = os.path.abspath(os.path.join(node_root, "..", ".."))
        candidates = [
            os.path.join(node_root, "prepare_config.json"),
            os.path.join(repo_root, "config", "comfy_prepare_config.json"),
            os.path.join(repo_root, "config", "prepare_config.json"),
            os.path.join(repo_root, "prepare_config.json"),
        ]

        found = None
        for c in candidates:
            if os.path.exists(c):
                found = c
                break

        if not found:
            raise FileNotFoundError(
                f"Required 'prepare_config.json' not found. Candidates searched: {candidates}"
            )

        # Load the found prepare_config (will raise if file missing/invalid)
        load_prepare_config(found)
        print(f"Loaded prepare config from {found}")

        model = ScorerModel(model_path)

        return ((model, maps),)


import types


class AestheticScoreNode:
    def __init__(self):
        # Defer heavy model initialization to first use (tests should not require
        # network downloads or extra native packages like SentencePiece).
        # Use a SimpleNamespace so tests can monkeypatch attributes like
        # `get_image_features` or `encode` without triggering heavy imports.
        self.siglip_model = types.SimpleNamespace()
        self.siglip_processor = types.SimpleNamespace()
        self.mpnet = types.SimpleNamespace()

    def _ensure_models_loaded(self):
        """Lazily load missing heavy model components if not provided by the caller.

        - If a test or caller has already provided a callable `get_image_features` on
          `siglip_model`, we won't replace that object (useful for unit tests).
        - Only load the specific components that are missing to avoid pulling in
          optional native deps (e.g., SentencePiece for tokenizers) when not
          required by the caller.
        """
        # Quick checks for provided callables
        siglip_ok = hasattr(self.siglip_model, "get_image_features") and callable(
            getattr(self.siglip_model, "get_image_features")
        )
        processor_ok = callable(self.siglip_processor)
        mpnet_ok = hasattr(self.mpnet, "encode") and callable(getattr(self.mpnet, "encode"))

        # If everything is provided by the caller, nothing to do
        if siglip_ok and processor_ok and mpnet_ok:
            return

        # Import heavy constructors only when needed
        from sentence_transformers import SentenceTransformer
        from transformers import AutoModel, AutoProcessor

        if not siglip_ok:
            self.siglip_model = AutoModel.from_pretrained(SIGLIP_ID)
        # Only load a processor if we're also loading the model; if the caller
        # provided a fake `get_image_features` we will bypass the processor
        # and avoid requiring tokenizers/SentencePiece in the environment.
        if (not processor_ok) and (not siglip_ok):
            self.siglip_processor = AutoProcessor.from_pretrained(SIGLIP_ID)
        if not mpnet_ok:
            self.mpnet = SentenceTransformer(MPNET_ID)

    def _prepare_image_batch(self, image: Any) -> list[Image.Image]:
        """
        Convert various image inputs into a list of RGB PIL Images.
        Accepts torch.Tensor, numpy arrays, PIL.Image, or list/tuple of these.
        Returns a list of `PIL.Image.Image` (RGB).
        """

        def array_to_pil(arr: Any) -> List[Image.Image]:
            arr = np.asarray(arr)
            if arr.ndim == 4:
                # Batch of images
                out: List[Image.Image] = []
                for i in range(arr.shape[0]):
                    sub = array_to_pil(arr[i])
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

        # Start handling different input types
        if isinstance(image, Image.Image):
            return [image.convert("RGB")]
        if isinstance(image, torch.Tensor):
            res = array_to_pil(image.detach().cpu().numpy())
            return res
        if isinstance(image, np.ndarray):
            res = array_to_pil(image)
            return res
        if isinstance(image, (list, tuple)):
            out: List[Image.Image] = []
            for it in image:
                out.extend(self._prepare_image_batch(it))
            return out
        raise TypeError(f"Unsupported image type: {type(image)}")

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "scorer": ("AESTHETIC_SCORER",),
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
        scorer: AestheticScorerLoader,
        image: Any,
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
    ) -> tuple[torch.Tensor, torch.Tensor, bool, List[float]]:

        batch_size = 10

        # Input validation for all fields (basic)
        if not (isinstance(scorer, tuple) and len(scorer) == 2):
            raise TypeError("'scorer' must be a tuple of (model, maps).")

        # Normalize incoming image(s) to a list of PIL.Images.
        try:
            images = self._prepare_image_batch(image)
        except Exception as e:
            raise TypeError(
                f"'image' could not be processed into a batch of images: {e}"
            )

        if not images:
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

        # scorer_instance: tuple[ScorerModel, Dict[str, Dict[str, int]]] = scorer
        scorer_model, maps = scorer
        maps: Dict[str, Dict[str, int]] = dict(maps)

        # --- Apply feature filtering and interaction features ---
        # Model path is e.g. .../models/bin, so use that as bin dir
        model_path = getattr(scorer_model, "model_path", None)
        if model_path is None:
            # Fallback: try to infer from ONNX path or parent dir
            model_path = getattr(scorer_model, "onnx_path", None)
        if model_path is None:
            # Fallback: use default relative to this file
            model_path = os.path.join(os.path.dirname(__file__), "models", "bin")
        if os.path.isfile(model_path):
            model_bin_dir = os.path.dirname(model_path)
        else:
            model_bin_dir = model_path

        # Ensure heavy models are loaded lazily (tests won't require sentencepiece/tokenizers)
        self._ensure_models_loaded()
        pos_vec: np.ndarray = np.asarray(self.mpnet.encode(positive))
        neg_vec: np.ndarray = np.asarray(self.mpnet.encode(negative))
        w, h = images[0].size  # Assume all same size in batch
        meta: Dict[str, Any] = {
            "steps": steps,
            "cfg": cfg,
            "width": w,
            "height": h,
            "lora_weight": lora_strength,
        }

        cat_indices = {
            "sampler": map_categorical_value(maps, "sampler", sampler),
            "scheduler": map_categorical_value(maps, "scheduler", scheduler),
            "model": map_categorical_value(maps, "model", model_name),
            "lora": map_categorical_value(maps, "lora", lora_name),
        }
        meta_vec = assemble_feature_vector(meta, pos_vec, neg_vec, cat_indices)

        all_scores: List[float] = []
        images_list: List[Any] = images

        # Process images in batches and collect scores for all images in original order
        idx = 0
        while idx < len(images_list):
            current_batch = images_list[idx : idx + batch_size]
            idx += batch_size

            full_vecs: List[np.ndarray] = []

            # If a real processor is available, use it to create model inputs.
            # In tests the `siglip_model` is often patched directly, so we allow
            # bypassing the processor and call `get_image_features` with the
            # raw images if needed.
            if callable(self.siglip_processor):
                inputs = self.siglip_processor(images=current_batch, return_tensors="pt")
                # Move tensors to the model device if the model exposes a `.device`
                try:
                    inputs = {k: v.to(self.siglip_model.device) for k, v in inputs.items()}
                except Exception:
                    # Best-effort; if moving to device fails, pass raw inputs to the model
                    pass
                with torch.no_grad():
                    vision_outputs = self.siglip_model.get_image_features(**inputs)
            else:
                # No processor available (likely a unit test stub); call model directly
                with torch.no_grad():
                    vision_outputs = self.siglip_model.get_image_features(images=current_batch)
            vision_vecs = vision_outputs.cpu().numpy().astype(float).tolist()

            for vec in vision_vecs:
                if len(vec) != IMAGE_VEC_LEN:
                    msg = f"CLIP returned unexpected vector length {len(vec)}, expected {IMAGE_VEC_LEN}"
                    raise RuntimeError(msg)

            for i in range(len(current_batch)):
                full_vecs.append(np.hstack([vision_vecs[i], meta_vec]))

            filtered_vecs = apply_feature_filter(full_vecs, model_bin_dir)
            final_vecs = apply_interaction_features(filtered_vecs, model_bin_dir)

            batch_scores = list(scorer_model.predict(final_vecs))
            all_scores.extend(batch_scores)

        # Now we have `images_list` and `all_scores` matching by index
        n_images = len(images_list)
        if len(all_scores) != n_images:
            raise RuntimeError("Scoring produced a different number of scores than images")

        # Sort images by score (descending)
        sorted_indices = sorted(range(n_images), key=lambda i: all_scores[i], reverse=True)

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
            create_image_batch(selected_images),
            create_image_batch(discarded_images),
            len(selected_images) > 0,
            all_scores,
        )


class TextScorerLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("TEXT_SCORER",)
    RETURN_NAMES = ("scorer",)
    FUNCTION = "load_model"
    CATEGORY = "Scoring"

    def load_model(self, model_path: str = "") -> tuple[tuple[ScorerModel, Dict[str, Dict[str, int]]]]:
        # Resolve model path - prefer internal text model
        if not model_path:
            internal_path = os.path.join(os.path.dirname(__file__), "models", "text", "bin")
            model_path = internal_path

        if not os.path.isabs(model_path):
            base = os.path.dirname(__file__)
            potential = os.path.join(base, model_path)
            if os.path.exists(potential):
                model_path = potential

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path not found: {model_path}")

        maps_path = os.path.join(os.path.dirname(__file__), "models", "maps")
        maps = load_maps(maps_path)

        # Require node-local config
        node_root = os.path.dirname(__file__)
        candidate = os.path.join(node_root, "prepare_config.json")
        if not os.path.exists(candidate):
            raise FileNotFoundError(
                f"Required 'prepare_config.json' not found in node folder: {candidate}"
            )
        load_prepare_config(candidate)

        model = ScorerModel(model_path)
        return ((model, maps),)


class TextScoreNode:
    """Scores text+params only using same vector assembly logic (no image features)."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "scorer": ("TEXT_SCORER",),
                "positive": ("STRING", {"multiline": True}),
                "negative": ("STRING", {"multiline": True}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 1000}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0}),
                "sampler": ("STRING", {"default": "euler"}),
                "scheduler": ("STRING", {"default": "normal"}),
                "model_name": ("STRING", {"default": "unknown"}),
                "lora_name": ("STRING", {"default": "unknown"}),
                "lora_strength": ("FLOAT", {"default": 0.0}),
                "width": ("INT", {"default": 512}),
                "height": ("INT", {"default": 512}),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("score",)
    FUNCTION = "calculate_score"
    CATEGORY = "Scoring"

    def __init__(self):
        # Lazy-load sentence transformer to avoid heavy imports at test time
        import types

        self.mpnet = types.SimpleNamespace()

    def _ensure_models_loaded(self):
        import types

        if isinstance(self.mpnet, types.SimpleNamespace):
            from sentence_transformers import SentenceTransformer

            self.mpnet = SentenceTransformer(MPNET_ID)

    def calculate_score(
        self,
        scorer: TextScorerLoader,
        positive: str,
        negative: str,
        steps: int,
        cfg: float,
        sampler: str,
        scheduler: str,
        model_name: str,
        lora_name: str,
        lora_strength: float,
        width: int,
        height: int,
    ) -> tuple[float]:
        if not (isinstance(scorer, tuple) and len(scorer) == 2):
            raise TypeError("'scorer' must be a tuple of (model, maps).")

        scorer_model, maps = scorer

        if not positive.strip():
            raise ValueError("The 'positive' prompt must be a non-empty string.")
        if not negative.strip():
            raise ValueError("The 'negative' prompt must be a non-empty string.")

        # Ensure mpnet is loaded lazily
        self._ensure_models_loaded()
        pos_vec = np.asarray(self.mpnet.encode(positive))
        neg_vec = np.asarray(self.mpnet.encode(negative))

        meta = {
            "steps": steps,
            "cfg": cfg,
            "width": width,
            "height": height,
            "lora_weight": lora_strength,
        }

        cat_indices = {
            "sampler": map_categorical_value(maps, "sampler", sampler),
            "scheduler": map_categorical_value(maps, "scheduler", scheduler),
            "model": map_categorical_value(maps, "model", model_name),
            "lora": map_categorical_value(maps, "lora", lora_name),
        }

        full_vec = assemble_feature_vector(meta, pos_vec, neg_vec, cat_indices)

        # Apply same filtering and interaction pipeline as image node expects
        model_path = getattr(scorer_model, "model_path", None)
        if model_path is None:
            model_path = getattr(scorer_model, "onnx_path", None)
        if model_path is None:
            model_bin_dir = os.path.join(os.path.dirname(__file__), "models", "bin")
        elif os.path.isfile(model_path):
            model_bin_dir = os.path.dirname(model_path)
        else:
            model_bin_dir = model_path

        filtered = apply_feature_filter([full_vec], model_bin_dir)
        final = apply_interaction_features(filtered, model_bin_dir)

        score = float(scorer_model.predict(final)[0])

        return (score,)


NODE_CLASS_MAPPINGS: Dict[str, Any] = {
    "AestheticScorerLoader": AestheticScorerLoader,
    "AestheticScoreNode": AestheticScoreNode,
    "TextScorerLoader": TextScorerLoader,
    "TextScoreNode": TextScoreNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AestheticScorerLoader": "Load Aesthetic Scorer",
    "AestheticScoreNode": "Calculate Aesthetic Score",
    "TextScorerLoader": "Load Text Scorer",
    "TextScoreNode": "Score Text+Params",
}
