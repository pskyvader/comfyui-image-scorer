from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from ..logger import get_logger

from .helpers import get_value_from_entry, l2_normalize_batch
from ..loaders.model_loader import model_loader

logger = get_logger(__name__)


class EmbeddingVector:
    def __init__(self, name: str, slot_size: int) -> None:
        self.name = name
        self.slot_size = slot_size
        self.value_list: dict[str, str] = {}
        self.vector_list: dict[str, list[float]] = {}
        self.text_list: dict[str, str] = {}

    def parse_value_list(
        self, entries: dict[str, dict[str, Any]], alias: list[str] | None = None
    ) -> dict[str, str]:
        self.value_list = {}
        for id, entry in list(entries.items()):
            current_value = get_value_from_entry(entry, self.name, alias)
            if not current_value:
                current_value = ""
            self.value_list[id] = str(current_value)
        result = self.value_list

        return result

    def create_vector_batch(
        self, current_batch: Iterable[tuple[str, str]]
    ) -> dict[str, list[float]]:
        model, vector_length = model_loader.load_embedding_model()
        batch_id, batch_values = zip(*current_batch)
        encoded_values = model.encode(list(batch_values))
        processed: npt.NDArray[np.float32] = np.asarray(
            encoded_values, dtype=np.float32
        )
        if processed.shape[-1] != vector_length:
            raise RuntimeError(
                "CLIP returned unexpected vector length "
                f"{processed.shape[-1]}, expected {vector_length}"
            )
        if vector_length != self.slot_size:
            raise RuntimeError(
                f"Embedding model output length {vector_length} for '{self.name}' "
                f"does not match configured slot_size {self.slot_size}"
            )
        normalized: npt.NDArray[np.float32] = l2_normalize_batch(processed)
        normalized_list = normalized.tolist()
        result: dict[str, list[float]] = dict(zip(batch_id, normalized_list))

        return result

    def create_vector_list(self, batch_size: int) -> dict[str, list[float]]:
        total = len(self.value_list.items())
        if total == 0:
            result: dict[str, list[float]] = {}
            return result

        with tqdm(
            total=total, desc="Encoded", unit=" " + self.name + " vectors", delay=3.0
        ) as pbar:
            for index in range(0, total, batch_size):
                current_batch: Iterable[tuple[str, str]] = list(
                    self.value_list.items()
                )[index : index + batch_size]
                batch_vecs: dict[str, list[float]] = self.create_vector_batch(
                    current_batch
                )
                self.vector_list.update(batch_vecs)
                pbar.update(len(current_batch))
        result = self.vector_list

        return result

    def create_text_batch(
        self, current_batch: Iterable[tuple[str, str]]
    ) -> dict[str, str]:
        batch_id, batch_values = zip(*current_batch)
        result: dict[str, str] = dict(zip(batch_id, batch_values))
        return result

    def create_text_list(self, batch_size: int) -> dict[str, str]:
        total = len(self.value_list.items())
        if total == 0:
            result: dict[str, str] = {}
            return result

        with tqdm(total=total, desc="Encoded", unit=" " + self.name + " texts", delay=3.0) as pbar:
            for index in range(0, total, batch_size):
                current_batch: Iterable[tuple[str, str]] = list(
                    self.value_list.items()
                )[index : index + batch_size]
                batch_vecs: dict[str, str] = self.create_text_batch(current_batch)
                self.text_list.update(batch_vecs)
                pbar.update(len(current_batch))
        result = self.text_list

        return result
