import torch
from typing import Dict, Any, List,Tuple
import numpy as np
from .model import ScorerModel
from ...shared.helpers import prepare_image_batch, export_image_batch, load_maps
from ...shared.feature_assembler import (
    map_categorical_value,
    apply_feature_filter,
    apply_interaction_features,
    assemble_feature_vector,
)
from ..paths import prepare_config
from ...shared.config import config


from sentence_transformers import SentenceTransformer

# from transformers import AutoModel, AutoProcessor
from transformers.models.siglip import modeling_siglip, processing_siglip

from ...shared.vectors.vectors import VectorList
from ...shared.image_analysis import ImageAnalysis
from ...shared.vectors.image_vector import ImageVector
class AestheticScoreNode:
    def __init__(self):
        SIGLIP_ID: str = prepare_config["vision_model"][
            "name"
        ]  # google/siglip-base-patch16-224
        MPNET_ID: str = prepare_config["prompt_representation"][
            "model"
        ]  # all-mpnet-base-v2

        self.siglip_model = modeling_siglip.SiglipModel.from_pretrained(SIGLIP_ID)
        self.siglip_processor = processing_siglip.SiglipProcessor.from_pretrained(
            SIGLIP_ID
        )
        self.mpnet = SentenceTransformer(MPNET_ID)

        self.model = ScorerModel()
        self.maps: Dict[str, Dict[str, int]] = load_maps()

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

        # # Normalize incoming image(s) to a list of PIL.Images.
        # try:
        #     images = prepare_image_batch(image)
        # except Exception as e:
        #     raise TypeError(
        #         f"'image' could not be processed into a batch of images: {e}"
        #     )

        if not image:
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

        entry:Dict[str,Any]={
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
        image_analysis=ImageAnalysis([])
        images_list=image_analysis.prepare_image_batch(image)
        processed_data:List[Tuple[str,Dict[str,Any],str,str]]=[]
        for i in range(0, len(images_list), batch_size):
            current_batch = images_list[i : i + batch_size]
            data:Tuple[str,Dict[str,Any],str,str]=("",entry,"","")
            data_batch:List[Tuple[str,Dict[str,Any],str,str]]=[data]*len(current_batch)
            data_batch=image_analysis.analyze_image_batch(current_batch,data_batch)
            processed_data.extend(data_batch)

        vector_list = VectorList(processed_data,[],[],[],add_new=False,merge_lists=False)
        vector_list.create_vectors()
        image_vector:ImageVector =vector_list.sorted_vectors["image"]["vector"]
        
        processed_images: List[List[float]]=[]
        for i in range(0, len(images_list), batch_size):
            current_batch = images_list[i : i + batch_size]
            current_processed_images=image_vector.create_image_vector_batch(current_batch)
            processed_images.extend(current_processed_images)

        image_vector.vector_list=processed_images
        vector_list.sorted_vectors["image"]["vector"]=image_vector

        final_vectors=vector_list.join_vectors()

        



        encode = self.mpnet.encode
        pos_vec: np.ndarray = np.asarray(encode(positive))
        neg_vec: np.ndarray = np.asarray(encode(negative))
        w, h = images[0].size  # Assume all same size in batch
        meta: Dict[str, Any] = {
            "steps": steps,
            "cfg": cfg,
            "original_width": w,
            "original_height": h,
            "final_width": w,
            "final_height": h,
            "lora_weight": lora_strength,
        }

        cat_indices = {
            "sampler": map_categorical_value(self.maps, "sampler", sampler),
            "scheduler": map_categorical_value(self.maps, "scheduler", scheduler),
            "model": map_categorical_value(self.maps, "model", model_name),
            "lora": map_categorical_value(self.maps, "lora", lora_name),
        }
        print(f"meta length: {len(meta)}")
        print(f"pos_vec length: {len(pos_vec)}")
        print(f"neg_vec length: {len(neg_vec)}")

        # same meta vector for all images
        meta_vec = assemble_feature_vector(meta, pos_vec, neg_vec, cat_indices)
        print(f"meta vector length: {len(meta_vec)}")

        all_scores: List[float] = []
        images_list: List[Any] = images

        # Process images in batches and collect scores for all images in original order
        idx = 0
        IMAGE_VEC_LEN = prepare_config["prompt_representation"]["dim"]
        while idx < len(images_list):
            current_batch = images_list[idx : idx + batch_size]
            idx += batch_size

            full_vecs: List[np.ndarray] = []

            inputs = self.siglip_processor(images=current_batch, return_tensors="pt")
            try:
                inputs = {k: v.to(self.siglip_model.device) for k, v in inputs.items()}
            except Exception:
                # Best-effort; if moving to device fails, pass raw inputs to the model
                pass
            with torch.no_grad():
                vision_outputs = self.siglip_model.get_image_features(**inputs)

            # vision_vecs = vision_outputs.cpu().numpy().astype(float).tolist()
            vision_vecs = (
                vision_outputs.pooler_output.cpu().numpy().astype(float).tolist()
            )

            for vec in vision_vecs:
                if len(vec) != IMAGE_VEC_LEN:
                    msg = f"CLIP returned unexpected vision vector length {len(vec)}, expected {IMAGE_VEC_LEN}"
                    raise RuntimeError(msg)

            for i in range(len(current_batch)):
                full_vecs.append(np.hstack([vision_vecs[i], meta_vec]))

            filtered_vecs = apply_feature_filter(full_vecs)
            final_vecs = apply_interaction_features(filtered_vecs)

            batch_scores = list(self.model.predict(final_vecs))
            all_scores.extend(batch_scores)

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
