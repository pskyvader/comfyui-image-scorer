from typing import Any
import numpy as np
from tqdm import tqdm
from .helpers import get_value_from_entry, l2_normalize_batch
from ..loaders.model_loader import model_loader
from ..vectors.terms import extract_terms


class EmbeddingVector:
    def __init__(self, name: str) -> None:
        self.name = name
        self.value_list: list[str] = []
        self.vector_list: list[list[float]] = []
        self.text_list: list[list[tuple[str, float]]] = []

    def parse_value_list(
        self, entries: list[dict[str, Any]], alias: list[str] | None = None
    ) -> list[str]:
        for entry in entries:
            # for entry_date in entry.values():
            current_value = get_value_from_entry(entry, self.name, alias)
            if not current_value:
                current_value = ""
            self.value_list.append(current_value)

        return self.value_list

    def create_vector_batch(self, current_batch: list[str]) -> list[list[float]]:

        model, vector_length = model_loader.load_embedding_model()

        # Encode entire batch at once (critical change)
        encoded_values = model.encode(current_batch)

        # Convert once to float32
        processed = np.asarray(encoded_values, dtype=np.float32)

        if processed.shape[-1] != vector_length:
            raise RuntimeError(
                f"CLIP returned unexpected vector length "
                f"{processed.shape[-1]}, expected {vector_length}"
            )

        normalized = l2_normalize_batch(processed)

        # Convert to Python lists only once at the end
        return normalized.tolist()

    def create_vector_list(self, batch_size: int = 4) -> list[list[float]]:
        total = len(self.value_list)

        with tqdm(total=total, desc="Encoded", unit=" " + self.name) as pbar:
            for i in range(0, total, batch_size):
                current_batch = self.value_list[i : i + batch_size]
                batch_vecs = self.create_vector_batch(current_batch)
                self.vector_list.extend(batch_vecs)
                pbar.update(len(current_batch))

        return self.vector_list

    def create_text_batch(self, batch: list[str]) -> list[list[tuple[str, float]]]:
        text_list: list[list[tuple[str, float]]] = []
        for text in batch:
            text_list.append(extract_terms(text))
        return text_list

    def create_text_list(self, batch_size: int = 4) -> list[list[tuple[str, float]]]:
        total = len(self.value_list)
        with tqdm(total=total, desc="Mapped", unit=" " + self.name) as pbar:
            for i in range(0, total, batch_size):
                current_batch = self.value_list[i : i + batch_size]
                batch_text = self.create_text_batch(current_batch)
                self.text_list.extend(batch_text)
                pbar.update(len(current_batch))
        return self.text_list
