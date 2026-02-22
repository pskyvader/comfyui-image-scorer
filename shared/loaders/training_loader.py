import numpy as np
import os
from typing import Any, Optional, Tuple, Dict
from pathlib import Path
from torch import nn
import pickle
import base64


from ..io import load_single_jsonl
from ..paths import (
    vectors_file,
    scores_file,
    filtered_data,
    models_dir,
    interaction_data,
    training_model,
)
from ..helpers import remove_directory


class TrainingLoader:
    def __init__(self, use_cache: bool):
        self.training_model: Optional[Any] = None
        self.processed_data: Optional[Any] = None
        self.filtered_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.interaction_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.vectors: Optional[np.ndarray] = None
        self.scores: Optional[np.ndarray] = None
        self.use_cache = use_cache

    def toggle_cache(self, use_cache: bool):
        self.use_cache = use_cache
        if not use_cache:
            self._reset_models()

    def _reset_models(self):
        self.vectors = None
        self.scores = None
        self.training_model = None
        self.processed_data = None
        self.filtered_data = None
        self.interaction_data = None

    def remove_training_models(self):
        remove_directory(Path(models_dir))
        self._reset_models()

    def load_vectors(self) -> np.ndarray:
        if self.use_cache and self.vectors:
            return self.vectors

        vectors = load_single_jsonl(vectors_file)
        x_vector: np.ndarray = np.array(vectors, dtype=float)
        self.vectors = x_vector
        return self.vectors

    def load_scores(self) -> np.ndarray:
        if self.use_cache and self.scores:
            return self.scores

        scores = load_single_jsonl(scores_file)
        y_vector: np.ndarray = np.array(scores, dtype=float)
        self.scores = y_vector
        return self.scores

    def load_filtered_data(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self.use_cache and self.filtered_data:
            return self.filtered_data

        if os.path.exists(filtered_data):
            try:
                data = np.load(filtered_data)
                if "X" in data and "kept_indices" in data:
                    # Validate shape compatibility if possible or just trust cache
                    self.filtered_data = data["X"], data["kept_indices"]
                    return self.filtered_data
            except Exception:
                pass
        return None

    def save_filtered_data(
        self, x: np.ndarray, kept_indices: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        os.makedirs(models_dir, exist_ok=True)
        np.savez_compressed(filtered_data, X=x, kept_indices=kept_indices)
        saved_data = (x, kept_indices)
        if self.use_cache:
            self.filtered_data = saved_data
        return saved_data

    def load_interaction_data(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self.use_cache and self.interaction_data:
            return self.interaction_data

        if os.path.exists(interaction_data):
            try:
                data = np.load(interaction_data)
                if "X" in data and "interaction_indices" in data:
                    print(f"Loading interaction data from cache: {interaction_data}")
                    self.interaction_data = data["X"], data["interaction_indices"]
                    return self.interaction_data
            except Exception:
                pass
        return None

    def save_interaction_data(
        self, x: np.ndarray, top_k_indices_local: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        os.makedirs(models_dir, exist_ok=True)
        np.savez_compressed(
            interaction_data, X=x, interaction_indices=top_k_indices_local
        )
        saved_data = (x, top_k_indices_local)
        if self.use_cache:
            self.interaction_data = saved_data
        return saved_data

    def _normalize(self, val: Any) -> Any:
        if isinstance(val, np.ndarray):
            if val.shape == ():
                return val.item()
            return val.copy()
        return val

    def load_training_model_diagnostics(self) -> Optional[Dict[str, Any]]:
        with np.load(Path(training_model), allow_pickle=True) as npz:
            return {k: self._normalize(npz[k]) for k in npz.files}

    def load_training_model(self):
        if self.training_model is not None:
            return self.training_model

        with np.load(Path(training_model), allow_pickle=True) as npz:
            if "__model_b64__" not in npz.files:
                raise KeyError(
                    f"No '__model_b64__' key found in {training_model}. Available keys: {list(npz.files)}"
                )
            model_b64 = npz["__model_b64__"].item()
            model_bytes = base64.b64decode(model_b64.encode("ascii"))

            self.training_model = pickle.loads(model_bytes)
        return self.training_model

    def save_training_model(
        self, model: Any, additional_data: Dict[str, Any] | None
    ) -> None:
        """Save a trained model to disk.

        Saves both the model and diagnostic data into a single .npz file.
        Encodes the model as base64 string to work with npz format.
        """
        os.makedirs(models_dir, exist_ok=True)

        # Pickle and encode the model to base64 (so it can be stored in npz)
        model_bytes = pickle.dumps(model)
        model_b64 = base64.b64encode(model_bytes).decode("ascii")

        # Create save data with encoded model
        save_data = {"__model_b64__": model_b64}
        if additional_data:
            save_data.update(additional_data)

        # Save everything to a single .npz file
        clean_data = {k: v for k, v in save_data.items() if v is not None}
        np.savez_compressed(training_model, **clean_data, allow_pickle=True)
        print(f"Saved model and diagnostics to: {training_model}")


training_loader = TrainingLoader(True)
