import threading
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image

from ..loaders.model_loader import model_loader

from ..logger import get_logger, ModuleLogger
logger: ModuleLogger = get_logger(__name__)

# Category order MUST match maps_loader.AGE/GENDER/RACE_CATEGORIES (model output index order).
AGE_LABELS = [
    "0-2",
    "10-19",
    "20-29",
    "3-9",
    "30-39",
    "40-49",
    "50-59",
    "60-69",
    "more than 70",
]
GENDER_LABELS = ["Female", "Male"]
RACE_LABELS = [
    "Black",
    "East Asian",
    "Indian",
    "Latino_Hispanic",
    "Middle Eastern",
    "Southeast Asian",
    "White",
]


class FaceAttributeAnalyzer:
    """Predicts perceived age, gender, and race from face images.

    Output is a list (one per input image) of dicts::

        {
            "age":    [ {label: prob}, ... ],   # one dict per detected person
            "gender": [ {label: prob}, ... ],
            "race":   [ {label: prob}, ... ],
        }

    Each per-person dict holds the softmax distribution over the fixed label set.
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

    def predict(self, img: Image.Image) -> dict[str, list[dict[str, float]]]:
        return self.predict_batch([img])[0]

    def predict_batch(
        self, imgs: Sequence[Image.Image]
    ) -> list[dict[str, list[dict[str, float]]]]:
        self._ensure_loaded()
        device = next(self._model.parameters()).device
        inputs = self._processor(images=list(imgs), return_tensors="pt").to(device)
        with torch.no_grad():
            logits = self._model(pixel_values=inputs["pixel_values"])

        age = logits["age"].cpu().float()
        gender = logits["gender"].cpu().float()
        race = logits["race"].cpu().float()

        # Normalize to (batch, persons, classes) — single-face models give (batch, classes).
        if age.ndim == 2:
            age = age[:, None, :]
        if gender.ndim == 2:
            gender = gender[:, None, :]
        if race.ndim == 2:
            race = race[:, None, :]

        age_p = F.softmax(age, dim=-1).numpy()
        gender_p = F.softmax(gender, dim=-1).numpy()
        race_p = F.softmax(race, dim=-1).numpy()

        results: list[dict[str, list[dict[str, float]]]] = []
        for b in range(age_p.shape[0]):
            age_dicts = [
                {AGE_LABELS[i]: float(age_p[b, p, i]) for i in range(age_p.shape[2])}
                for p in range(age_p.shape[1])
            ]
            gender_dicts = [
                {GENDER_LABELS[i]: float(gender_p[b, p, i]) for i in range(gender_p.shape[2])}
                for p in range(gender_p.shape[1])
            ]
            race_dicts = [
                {RACE_LABELS[i]: float(race_p[b, p, i]) for i in range(race_p.shape[2])}
                for p in range(race_p.shape[1])
            ]
            results.append(
                {"age": age_dicts, "gender": gender_dicts, "race": race_dicts}
            )
        return results


class NSFWAnalyzer:
    """Predicts NSFW probability for an image using a ViT classifier."""

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
