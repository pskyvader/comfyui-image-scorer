from __future__ import annotations

import time
from typing import Any

import numpy as np
from tqdm import tqdm

from ..logger import get_logger

from .helpers import get_value_from_entry, l2_normalize_batch
from ..loaders.model_loader import model_loader

logger = get_logger(__name__)


class EmbeddingVector:
    def __init__(self, name: str) -> None:
        _start = time.perf_counter()
        self.name = name
        self.value_list: list[str] = []
        self.vector_list: list[list[float]] = []
        self.text_list: list[str] = []

    def parse_value_list(
        self, entries: list[dict[str, Any]], alias: list[str] | None = None
    ) -> list[str]:
        _start = time.perf_counter()
        self.value_list = []
        for entry in entries:
            current_value = get_value_from_entry(entry, self.name, alias)
            if not current_value:
                current_value = ""
            self.value_list.append(str(current_value))
        result = self.value_list

        return result

    def create_vector_batch(self, current_batch: list[str]) -> list[list[float]]:
        _start = time.perf_counter()
        model, vector_length = model_loader.load_embedding_model()
        encoded_values = model.encode(current_batch)
        processed = np.asarray(encoded_values, dtype=np.float32)
        if processed.shape[-1] != vector_length:
            raise RuntimeError(
                "CLIP returned unexpected vector length "
                f"{processed.shape[-1]}, expected {vector_length}"
            )
        normalized = l2_normalize_batch(processed)
        result = normalized.tolist()

        return result

    def create_vector_list(self, batch_size: int = 4) -> list[list[float]]:
        _start = time.perf_counter()
        total = len(self.value_list)
        if total == 0:
            result: list[list[float]] = []

            return result

        with tqdm(total=total, desc="Encoded", unit=" " + self.name) as pbar:
            for index in range(0, total, batch_size):
                current_batch = self.value_list[index : index + batch_size]
                batch_vecs = self.create_vector_batch(current_batch)
                self.vector_list.extend(batch_vecs)
                pbar.update(len(current_batch))
        result = self.vector_list

        return result

    def create_text_batch(self, batch: list[str]) -> list[str]:
        _start = time.perf_counter()
        result = batch

        return result

    def create_text_list(self, batch_size: int = 4) -> list[str]:
        _start = time.perf_counter()
        total = len(self.value_list)
        if total == 0:
            result: list[str] = []

            return result

        with tqdm(total=total, desc="Mapped", unit=" " + self.name) as pbar:
            for index in range(0, total, batch_size):
                current_batch = self.value_list[index : index + batch_size]
                batch_text = self.create_text_batch(current_batch)
                self.text_list.extend(batch_text)
                pbar.update(len(current_batch))
        result = self.text_list

        return result
