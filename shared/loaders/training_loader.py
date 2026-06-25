import numpy as np
from numpy import typing as npt
import os
from typing import Any, Iterator
from pathlib import Path
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
    raw_data,
    index_file,
    comparisons_file,
    comparison_data,
)
from ..helpers import remove_directory
from ..logger import get_logger, ModuleLogger

logger: ModuleLogger = get_logger(__name__)


class TrainingLoader:
    def __init__(self, use_cache: bool):
        self.training_model: Any | None = None
        self.raw_data: (
            tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]] | None
        ) = None
        self.comparison_data: (
            tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]] | None
        ) = None
        self.filtered_data: (
            tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]] | None
        ) = None
        self.interaction_data: (
            tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]] | None
        ) = None
        self.vectors: npt.NDArray[np.float32] | None = None
        self.scores: npt.NDArray[np.float32] | None = None
        self.comparisons: dict[str, list[dict[str, str]]] | None = None
        self.use_cache = use_cache

    def _reset_models(self) -> None:
        self.vectors = None
        self.scores = None
        self.training_model = None
        self.raw_data = None
        self.comparison_data = None
        self.filtered_data = None
        self.interaction_data = None

    def remove_training_models(self) -> None:
        remove_directory(Path(models_dir))
        self._reset_models()

    def load_vectors(self) -> npt.NDArray[np.float32]:
        if self.use_cache and self.vectors is not None:
            return self.vectors

        logger.info("loading vectors file...")

        vectors = list(load_single_jsonl(vectors_file))
        x_vector: npt.NDArray[np.float32] = np.array(vectors, dtype=float)
        self.vectors = x_vector
        return self.vectors

    def load_scores(self) -> npt.NDArray[np.float32]:
        if self.use_cache and self.scores is not None:
            return self.scores
        logger.debug("loading scores file....")
        scores = list(load_single_jsonl(scores_file))
        y_vector: npt.NDArray[np.float32] = np.array(scores, dtype=float)
        self.scores = y_vector
        return self.scores

    def load_comparison_counts(self) -> dict[str, list[dict[str, str]]]:
        if self.use_cache and self.comparisons is not None:
            return self.comparisons
        logger.debug("loading index file....")
        index = load_single_jsonl(index_file)
        comparisons: Iterator[Any] = load_single_jsonl(comparisons_file)

        comparisons_count: dict[str, list[dict[str, str]]] = {}
        for i in index:
            comparisons_count[i] = []

        for c in comparisons:
            filename_a: str = c["filename_a"]
            filename_b: str = c["filename_b"]
            winner: str = c["winner"]

            record_a = {"other": filename_b, "winner": winner}
            record_b = {"other": filename_a, "winner": winner}
            if filename_a in comparisons_count:
                comparisons_count[filename_a].append(record_a)
            if filename_b in comparisons_count:
                comparisons_count[filename_b].append(record_b)
        self.comparisons = comparisons_count
        return self.comparisons

    def load_raw_data(
        self,
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]] | None:
        if self.use_cache and self.raw_data is not None:
            return self.raw_data

        if os.path.exists(raw_data):
            try:
                data = np.load(raw_data)
                if "x" in data and "y" in data:
                    self.raw_data = data["x"], data["y"]
                    return self.raw_data
            except Exception:
                pass
        return None

    def save_raw_data(
        self, x: npt.NDArray[np.float32], y: npt.NDArray[Any]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[Any]]:
        logger.debug("saving raw data...")
        os.makedirs(models_dir, exist_ok=True)
        np.savez_compressed(raw_data, x=x, y=y)
        saved_data = (x, y)
        if self.use_cache:
            self.raw_data = saved_data
        return saved_data

    def load_comparison_data(
        self,
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]] | None:
        if self.use_cache and self.comparison_data is not None:
            return self.comparison_data

        if os.path.exists(comparison_data):
            try:
                data = np.load(comparison_data)
                if "x" in data and "y" in data:
                    self.comparison_data = data["x"], data["y"]
                    return self.comparison_data
            except Exception:
                pass
        return None

    def save_comparison_data(
        self, x: npt.NDArray[np.float32], y: npt.NDArray[Any]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[Any]]:
        logger.debug("saving comparison data...")
        os.makedirs(models_dir, exist_ok=True)
        np.savez_compressed(comparison_data, x=x, y=y)
        saved_data = (x, y)
        if self.use_cache:
            self.comparison_data = saved_data
        return saved_data

    def load_filtered_data(
        self,
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]] | None:
        if self.use_cache and self.filtered_data is not None:
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
        self, x: npt.NDArray[np.float32], kept_indices: npt.NDArray[Any]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[Any]]:
        logger.debug("saving filtered data...")
        os.makedirs(models_dir, exist_ok=True)
        np.savez_compressed(filtered_data, X=x, kept_indices=kept_indices)
        saved_data = (x, kept_indices)
        if self.use_cache:
            self.filtered_data = saved_data
        return saved_data

    def load_interaction_data(
        self,
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]] | None:
        if self.use_cache and self.interaction_data is not None:
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
        self, x: npt.NDArray[np.float32], top_k_indices_local: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
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

    def load_training_model_diagnostics(self) -> dict[str, Any] | None:
        with np.load(Path(training_model), allow_pickle=True) as npz:
            return {k: self._normalize(npz[k]) for k in npz.files}

    def load_training_model(self) -> Any:
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
        self, model: Any, additional_data: dict[str, Any] | None
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
