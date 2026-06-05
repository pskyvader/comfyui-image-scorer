from typing import Any
import torch
from torch import nn
from torchvision import transforms
import timm
from ..config import config
from sentence_transformers import SentenceTransformer


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

        model = model.eval()  # type: ignore[union-attr]
        model.to(device)  # type: ignore[union-attr]

        print(f"Vision model '{model_key}' loaded on device: {device} ")

        props = torch.cuda.get_device_properties(device)  # type: ignore[union-attr]
        total_memory = int(props.total_memory)

        data_config = timm.data.resolve_model_data_config(model)
        input_size = data_config["input_size"]  # (3, H, W)
        model_input_size = (input_size[2], input_size[1])  # (W, H)

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
        # mode: str = embedding_config["mode"]

        if device != "cuda":
            raise RuntimeError("`clip_device` not set to 'cuda'")

        try:
            model = SentenceTransformer(name, device=device, local_files_only=True)
        except:
            model = SentenceTransformer(name, device=device)

        self.embedding_model = (model, output_dim)
        return self.embedding_model


# Global singleton
model_loader = ModelLoader()
