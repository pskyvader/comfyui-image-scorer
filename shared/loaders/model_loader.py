import os
import threading
from typing import Any
import torch
from torch import nn
from safetensors.torch import load_file as load_safetensors
from torchvision import transforms
import timm
from ..config import config
from sentence_transformers import SentenceTransformer
from transformers import AutoModel
from transformers import CLIPImageProcessor, AutoImageProcessor, AutoModelForImageClassification


class MultiTaskClipVisionModel(nn.Module):
    def __init__(self, num_labels: dict[str, int]) -> None:
        super().__init__()
        self.vision_model = AutoModel.from_pretrained(
            "openai/clip-vit-large-patch14"
        ).vision_model
        hidden_size = self.vision_model.config.hidden_size
        self.age_head = nn.Linear(hidden_size, num_labels["age"])
        self.gender_head = nn.Linear(hidden_size, num_labels["gender"])
        self.race_head = nn.Linear(hidden_size, num_labels["race"])

    def forward(
        self, pixel_values: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        outputs = self.vision_model(pixel_values=pixel_values)
        pooled = outputs.pooler_output
        return {
            "age": self.age_head(pooled),
            "gender": self.gender_head(pooled),
            "race": self.race_head(pooled),
        }


class ModelLoader:
    _IMAGENET_NORM = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    _CLIP_NORM = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )

    def __init__(self):
        self.embedding_model: tuple[SentenceTransformer, int] | None = None
        self.vision_model_cache: dict[
            str, tuple[nn.Module, int, int, transforms.Compose]
        ] = {}
        self._model_info_cache: dict[str, dict[str, Any]] = {}
        self._hf_model_cache: dict[str, tuple[nn.Module, int, Any]] = {}
        self._hf_model_lock = threading.Lock()
        self.cnn_model: Any | None = None
        self.prepare_config = config["prepare"]

    @staticmethod
    def _select_transform(name: str) -> transforms.Compose:
        if "clip" in name.lower():
            return ModelLoader._CLIP_NORM
        return ModelLoader._IMAGENET_NORM

    def load_vision_model(
        self, model_key: str = "convnext"
    ) -> tuple[nn.Module, int, int, transforms.Compose]:
        cached = self.vision_model_cache.get(model_key)
        if cached is not None:
            return cached

        vision_models: dict = self.prepare_config["vision_models"]
        if model_key not in vision_models:
            raise KeyError(
                f"Vision model key '{model_key}' not found in prepare_config. "
                f"Available: {list(vision_models.keys())}"
            )

        model_config = vision_models[model_key]
        device: str = model_config["device"]
        name: str = model_config["name"]
        output_dim: int = model_config["output_dim"]
        variable_input: bool = model_config["variable_input"]
        global_pool: str = model_config["global_pool"]

        if device != "cuda":
            raise RuntimeError("device not set to 'cuda'")

        print(f"Loading Vision Model ({model_key}): {name}...")

        model: nn.Module = timm.create_model(
            name,
            pretrained=True,
            num_classes=0,
            global_pool=global_pool,
        )

        model = model.eval()
        model.to(device)

        print(f"Vision model '{model_key}' loaded on device: {device} ")

        props = torch.cuda.get_device_properties(device)
        total_memory = int(props.total_memory)

        data_config = timm.data.resolve_model_data_config(model)
        input_size = data_config["input_size"]
        model_input_size = (input_size[2], input_size[1])

        transform = self._select_transform(name)
        if not variable_input:
            transform = transforms.Compose(
                [
                    transforms.Resize(model_input_size),
                    transform,
                ]
            )

        result = (model, output_dim, total_memory, transform)
        self.vision_model_cache[model_key] = result
        self._model_info_cache[model_key] = {
            "variable_input": variable_input,
            "input_size": model_input_size,
        }
        return result

    def get_model_info(self, model_key: str) -> dict[str, Any]:
        if model_key not in self.vision_model_cache:
            self.load_vision_model(model_key)
        return self._model_info_cache.get(model_key, {})

    def load_embedding_model(self) -> tuple[SentenceTransformer, int]:
        if self.embedding_model is not None:
            return self.embedding_model

        embedding_config = self.prepare_config["prompt_representation"]
        name: str = embedding_config["name"]
        output_dim: int = embedding_config["output_dim"]
        device: str = embedding_config["device"]

        if device != "cuda":
            raise RuntimeError("`clip_device` not set to 'cuda'")

        try:
            model = SentenceTransformer(name, device=device, local_files_only=True)
        except:
            model = SentenceTransformer(name, device=device)

        self.embedding_model = (model, output_dim)
        return self.embedding_model

    def load_hf_vision_model(self, model_key: str) -> tuple[nn.Module, int, Any]:
        cached = self._hf_model_cache.get(model_key)
        if cached is not None:
            return cached

        with self._hf_model_lock:
            cached = self._hf_model_cache.get(model_key)
            if cached is not None:
                return cached
            result = self._load_hf_vision_model_impl(model_key)

        self._hf_model_cache[model_key] = result
        return result

    def _load_hf_vision_model_impl(self, model_key: str) -> tuple[nn.Module, int, Any]:
        attribute_models: dict = self.prepare_config["attribute_models"]
        if model_key not in attribute_models:
            raise KeyError(
                f"Attribute model key '{model_key}' not found in prepare_config. "
                f"Available: {list(attribute_models.keys())}"
            )

        model_config = attribute_models[model_key]
        name: str = model_config["name"]
        output_dim: int = model_config["output_dim"]
        device: str = model_config["device"]

        print(f"Loading Attribute Model ({model_key}): {name}...")

        if model_key == "face_attributes":
            processor = CLIPImageProcessor.from_pretrained(name)
            num_labels = {"age": 9, "gender": 2, "race": 7}
            model = MultiTaskClipVisionModel(num_labels=num_labels)
            cache_dir = os.path.join(torch.hub.get_dir(), "checkpoints")
            os.makedirs(cache_dir, exist_ok=True)
            safe_name = name.replace("/", "_")
            cache_path = os.path.join(cache_dir, f"{safe_name}.safetensors")
            if not os.path.exists(cache_path):
                torch.hub.download_url_to_file(
                    f"https://huggingface.co/{name}/resolve/main/model.safetensors",
                    cache_path,
                )
            state_dict = load_safetensors(cache_path)
            model.load_state_dict(state_dict, strict=False)
            model = model.eval()
            model.to(device)
            print(f"Attribute model '{model_key}' loaded on device: {device}")
            result = (model, output_dim, processor)
        elif model_key == "nsfw":
            processor = AutoImageProcessor.from_pretrained(name)
            model = AutoModelForImageClassification.from_pretrained(name)
            model = model.eval()
            model.to(device)
            print(f"NSFW model '{model_key}' loaded on device: {device}")
            result = (model, output_dim, processor)
        else:
            raise KeyError(f"Unknown attribute model key: {model_key}")

        return result


model_loader = ModelLoader()
