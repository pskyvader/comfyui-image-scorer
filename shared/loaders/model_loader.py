from typing import Any, Optional, Tuple, Dict
import torch
from torch import nn
import timm
from ..config import config
from sentence_transformers import SentenceTransformer




class ModelLoader:
    def __init__(self):
        self.embedding_model: Optional[Tuple[Any, int]] = None
        self.vision_model: Optional[Tuple[nn.Module, int,int]] = None
        self.cnn_model = None
        self.filtered_data = None
        self.interaction_data = None
        self.trained_model = None
        self.prepare_config = config["prepare"]
        #torch.backends.cudnn.benchmark = True



    def load_vision_model(self) -> Tuple[nn.Module, int,int]:
        if self.vision_model is not None:
            return self.vision_model

        vision_model_config = self.prepare_config["vision_model"]

        device: str = vision_model_config["device"]
        name: str = vision_model_config["name"]
        output_dim: int = vision_model_config["output_dim"]

        if device != "cuda":
            raise RuntimeError("`clip_device` not set to 'cuda'")

        print(f"Loading Vision Model: {name}...")

        model: nn.Module = timm.create_model(
            name,
            pretrained=True,
            num_classes=0,
            global_pool="avg",
        )

        # Remove classifier head safely
        # if hasattr(model, "head"):
        #     model.head = nn.Identity()
        # elif hasattr(model, "fc"):
        #     model.fc = nn.Identity()

        model = model.eval()
        model.to(device)

        print(f"Vision model loaded on device: {device} ")

        # No processor for ConvNeXt
        props = torch.cuda.get_device_properties(device)
        total_memory= int(props.total_memory)
        
        self.vision_model = (model, output_dim,total_memory)
        return self.vision_model

    def load_embedding_model(self) -> Tuple[Any, int]:
        if self.embedding_model is not None:
            return self.embedding_model

        embedding_config = self.prepare_config["prompt_representation"]
        name: str = embedding_config["name"]
        output_dim: int = embedding_config["output_dim"]
        device: str = embedding_config["device"]
        mode: str = embedding_config["mode"]

        if device != "cuda":
            raise RuntimeError("`clip_device` not set to 'cuda'")

        model = SentenceTransformer(name, device=device)

        self.embedding_model = (model, output_dim)
        return self.embedding_model


# Global singleton
model_loader = ModelLoader()
