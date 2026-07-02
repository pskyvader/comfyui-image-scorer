import threading
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image

from ..loaders.model_loader import model_loader

from ..logger import get_logger, ModuleLogger
logger: ModuleLogger = get_logger(__name__)


class FaceAttributeAnalyzer:
    """
    Predicts perceived age, gender, and race from a face image
    using a multi-task CLIP model fine-tuned on FairFace.

    Internal structure of face_logits (18 values per person):
        [0..8]   age logits (9 classes)
                    0: 0-2, 1: 10-19, 2: 20-29, 3: 3-9,
                    4: 30-39, 5: 40-49, 6: 50-59,
                    7: 60-69, 8: more than 70
        [9..10]  gender logits (2 classes)
                    9: Female, 10: Male
        [11..17] race logits (7 classes)
                    11: Black, 12: East Asian, 13: Indian,
                    14: Latino_Hispanic, 15: Middle Eastern,
                    16: Southeast Asian, 17: White
    """

    MODEL_KEY = "face_attributes"

    def __init__(self) -> None:
        self._model: Any = None
        self._output_dim: int = 0
        self._processor: Any = None
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            model, output_dim, processor = model_loader.load_hf_vision_model(self.MODEL_KEY)
            self._model = model
            self._output_dim = output_dim
            self._processor = processor

    def predict(self, img: Image.Image) -> list[float]:
        return self.predict_batch([img])[0]

    def predict_batch(self, imgs: Sequence[Image.Image]) -> list[list[float]]:
        self._ensure_loaded()
        device = next(self._model.parameters()).device
        inputs = self._processor(images=list(imgs), return_tensors="pt").to(device)
        with torch.no_grad():
            logits = self._model(pixel_values=inputs["pixel_values"])
        age = logits["age"].cpu().float().numpy().tolist()
        gender = logits["gender"].cpu().float().numpy().tolist()
        race = logits["race"].cpu().float().numpy().tolist()
        return [a + g + r for a, g, r in zip(age, gender, race)]


class NSFWAnalyzer:
    """
    Predicts NSFW probability for an image using a ViT classifier.

    Internal structure of nsfw_score (float per image):
        [0]  probability of "nsfw" class (0-1)
    """

    MODEL_KEY = "nsfw"

    def __init__(self) -> None:
        self._model: Any = None
        self._output_dim: int = 0
        self._processor: Any = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        model, output_dim, processor = model_loader.load_hf_vision_model(self.MODEL_KEY)
        self._model = model
        self._output_dim = output_dim
        self._processor = processor

    def predict(self, img: Image.Image) -> float:
        return self.predict_batch([img])[0]

    def predict_batch(self, imgs: Sequence[Image.Image]) -> list[float]:
        self._ensure_loaded()
        device = next(self._model.parameters()).device
        inputs = self._processor(images=list(imgs), return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)
        nsfw_idx = 1
        return probs[:, nsfw_idx].cpu().float().numpy().tolist()
