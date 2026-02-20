from typing import List, Dict, Any
import numpy as np
from tqdm import tqdm
from .helpers import get_value_from_entry
from ..loaders.model_loader import model_loader


class EmbeddingVector:
    def __init__(self, name: str) -> None:
        self.name = name
        self.value_list: List[str] = []
        self.vector_list: List[List[float]] = []

    def parse_value_list(self, entries: List[Dict[str, Any]]) -> List[str]:
        for entry in entries:
            # for entry_date in entry.values():
            current_value = get_value_from_entry(entry, self.name)
            if not current_value:
                current_value = ""
            self.value_list.append(current_value)
        return self.value_list

    def create_vector_batch(self, current_batch: List[str]) -> List[List[float]]:

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

        # Convert to Python lists only once at the end
        return processed.tolist()

    def create_vector_list(self, batch_size: int = 4) -> List[List[float]]:
        total = len(self.value_list)

        with tqdm(total=total, desc="Encoded", unit=" " + self.name) as pbar:
            for i in range(0, total, batch_size):
                current_batch = self.value_list[i : i + batch_size]
                batch_vecs = self.create_vector_batch(current_batch)
                self.vector_list.extend(batch_vecs)
                pbar.update(len(current_batch))

        return self.vector_list
