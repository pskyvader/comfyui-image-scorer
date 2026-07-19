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
    models_dir,
    interaction_data,
    training_model,
    vectors_data,
    scores_data,
    comparisons_data,
    feature_rule,
    comparison_rule,
    index_file,
    comparisons_file,
)
from ..helpers import remove_directory
from ..logger import get_logger, ModuleLogger

logger: ModuleLogger = get_logger(__name__)


class TrainingLoader:
    def __init__(self, use_cache: bool):
        self.training_model: Any | None = None
        self.interaction_data: (
            tuple[npt.NDArray[np.float32], npt.NDArray[np.intp]] | None
        ) = None
        self.vectors: npt.NDArray[np.float32] | None = None
        self.scores: npt.NDArray[np.float32] | None = None
        self.vectors_keyed: dict[str, npt.NDArray[np.float32]] | None = None
        self.scores_keyed: dict[str, float] | None = None
        self.comparison_rows: list[dict[str, Any]] | None = None
        self.feature_rule: npt.NDArray[np.intp] | None = None
        self.comparison_rule: tuple[int, dict[str, tuple[float, int]]] | None = None
        self.use_cache = use_cache

    def _reset_models(self) -> None:
        self.vectors = None
        self.scores = None
        self.vectors_keyed = None
        self.scores_keyed = None
        self.comparison_rows = None
        self.training_model = None
        self.feature_rule = None
        self.comparison_rule = None
        self.interaction_data = None

    def remove_training_models(self) -> None:
        remove_directory(Path(models_dir))
        self._reset_models()

    def load_vectors(self) -> dict[str, npt.NDArray[np.float32]]:
        """Load vectors keyed by filename.

        Three-tier resolution: in-memory cache -> vectors.npz ->
        vectors.jsonl (then cached as vectors.npz). Raises if the jsonl source
        is absent.
        """
        if self.use_cache and self.vectors_keyed is not None:
            return self.vectors_keyed
        cached = self._load_vectors_from_npz()
        if cached is not None:
            if self.use_cache:
                self.vectors_keyed = cached
            return cached
        if not os.path.exists(vectors_file):
            raise FileNotFoundError(
                "Missing source jsonl file required to load vectors. "
                f"Expected {vectors_file}. Run the prepare pipeline "
                "(prepare_data) before training."
            )
        logger.info("loading vectors file...")
        keyed = self._load_vectors_from_jsonl()
        self._save_vectors_to_npz(keyed)
        if self.use_cache:
            self.vectors_keyed = keyed
        return keyed

    def load_vectors_array(self) -> npt.NDArray[np.float32]:
        if self.use_cache and self.vectors is not None:
            return self.vectors
        keyed = self.load_vectors()
        x_vector: npt.NDArray[np.float32] = np.array(
            [keyed[fid] for fid in keyed], dtype=np.float32
        )
        self.vectors = x_vector
        return self.vectors

    def _load_vectors_from_jsonl(self) -> dict[str, npt.NDArray[np.float32]]:
        keyed: dict[str, npt.NDArray[np.float32]] = {}
        for rec in load_single_jsonl(vectors_file):
            if not isinstance(rec, dict) or len(rec) != 1:
                continue
            fid, vec = next(iter(rec.items()))
            keyed[str(fid)] = np.asarray(vec, dtype=np.float32)
        return keyed

    def _load_vectors_from_npz(
        self,
    ) -> dict[str, npt.NDArray[np.float32]] | None:
        if os.path.exists(vectors_data):
            try:
                data = np.load(vectors_data, allow_pickle=True)
                if "x" in data and "keys" in data:
                    keys = list(data["keys"])
                    x_array = data["x"]
                    return {
                        str(keys[i]): x_array[i] for i in range(len(keys))
                    }
            except Exception:
                pass
        return None

    def _save_vectors_to_npz(
        self, keyed: dict[str, npt.NDArray[np.float32]]
    ) -> None:
        logger.debug("saving vectors cache...")
        os.makedirs(models_dir, exist_ok=True)
        order = list(keyed.keys())
        x_array = np.array([keyed[fid] for fid in order], dtype=np.float32)
        keys_array = np.array(order, dtype=object)
        np.savez_compressed(vectors_data, x=x_array, keys=keys_array)

    def load_scores(self) -> dict[str, float]:
        """Load scores keyed by filename.

        Scores are the single source of truth for training rows. Three-tier
        resolution: in-memory cache -> scores.npz -> scores.jsonl (then cached
        as scores.npz). Raises if the jsonl source is absent.
        """
        if self.use_cache and self.scores_keyed is not None:
            return self.scores_keyed
        cached = self._load_scores_from_npz()
        if cached is not None:
            if self.use_cache:
                self.scores_keyed = cached
            return cached
        if not os.path.exists(scores_file):
            raise FileNotFoundError(
                "Missing source jsonl file required to load scores. "
                f"Expected {scores_file}. Run the prepare pipeline "
                "(prepare_data) before training."
            )
        logger.debug("loading scores file....")
        keyed = self._load_scores_from_jsonl()
        self._save_scores_to_npz(keyed)
        if self.use_cache:
            self.scores_keyed = keyed
        return keyed

    def load_scores_array(self) -> npt.NDArray[np.float32]:
        if self.use_cache and self.scores is not None:
            return self.scores
        keyed = self.load_scores()
        y_vector: npt.NDArray[np.float32] = np.array(
            [keyed[fid] for fid in keyed], dtype=np.float32
        )
        self.scores = y_vector
        return self.scores

    def _load_scores_from_jsonl(self) -> dict[str, float]:
        keyed: dict[str, float] = {}
        for rec in load_single_jsonl(scores_file):
            if not isinstance(rec, dict) or len(rec) != 1:
                continue
            fid, score = next(iter(rec.items()))
            keyed[str(fid)] = float(score)
        return keyed

    def _load_scores_from_npz(self) -> dict[str, float] | None:
        if os.path.exists(scores_data):
            try:
                data = np.load(scores_data, allow_pickle=True)
                if "y" in data and "keys" in data:
                    keys = list(data["keys"])
                    y_array = data["y"]
                    return {
                        str(keys[i]): float(y_array[i])
                        for i in range(len(keys))
                    }
            except Exception:
                pass
        return None

    def _save_scores_to_npz(self, keyed: dict[str, float]) -> None:
        logger.debug("saving scores cache...")
        os.makedirs(models_dir, exist_ok=True)
        order = list(keyed.keys())
        y_array = np.array([keyed[fid] for fid in order], dtype=np.float32)
        keys_array = np.array(order, dtype=object)
        np.savez_compressed(scores_data, y=y_array, keys=keys_array)

    def load_comparison_rows(self) -> list[dict[str, Any]]:
        """Load ordered comparison rows (filename_a, filename_b, winner, id).

        Three-tier resolution: in-memory cache -> comparisons.npz ->
        comparisons.jsonl (then cached as comparisons.npz). Rows are the source
        of truth for both counts and rating replay, so order-carrying fields are
        preserved. Raises if the jsonl source is absent.
        """
        if self.use_cache and self.comparison_rows is not None:
            return self.comparison_rows
        cached = self._load_comparisons_from_npz()
        if cached is not None:
            if self.use_cache:
                self.comparison_rows = cached
            return cached
        if not os.path.exists(comparisons_file):
            raise FileNotFoundError(
                "Missing source jsonl file required to load comparisons. "
                f"Expected {comparisons_file}. Run the prepare pipeline "
                "(prepare_data) before training."
            )
        logger.debug("loading comparisons file....")
        rows = [
            {
                "id": int(c.get("comparison_id", 0) or 0),
                "filename_a": str(c["filename_a"]),
                "filename_b": str(c["filename_b"]),
                "winner": str(c["winner"]),
            }
            for c in load_single_jsonl(comparisons_file)
        ]
        self._save_comparisons_to_npz(rows)
        if self.use_cache:
            self.comparison_rows = rows
        return rows

    def load_comparison_counts(self) -> dict[str, int]:
        """Return per-filename comparison counts derived from the rows."""
        counts: dict[str, int] = {}
        for row in self.load_comparison_rows():
            counts[row["filename_a"]] = counts.get(row["filename_a"], 0) + 1
            counts[row["filename_b"]] = counts.get(row["filename_b"], 0) + 1
        return counts

    def _load_comparisons_from_npz(self) -> list[dict[str, Any]] | None:
        if os.path.exists(comparisons_data):
            try:
                data = np.load(comparisons_data, allow_pickle=True)
                if all(k in data for k in ("ids", "winners", "a", "b")):
                    ids = data["ids"]
                    winners = data["winners"]
                    a = data["a"]
                    b = data["b"]
                    return [
                        {
                            "id": int(ids[i]),
                            "filename_a": str(a[i]),
                            "filename_b": str(b[i]),
                            "winner": str(winners[i]),
                        }
                        for i in range(len(ids))
                    ]
            except Exception:
                pass
        return None

    def _save_comparisons_to_npz(self, rows: list[dict[str, Any]]) -> None:
        logger.debug("saving comparisons cache...")
        os.makedirs(models_dir, exist_ok=True)
        ids = np.array([r["id"] for r in rows], dtype=np.int64)
        winners = np.array([r["winner"] for r in rows], dtype=object)
        a = np.array([r["filename_a"] for r in rows], dtype=object)
        b = np.array([r["filename_b"] for r in rows], dtype=object)
        np.savez_compressed(comparisons_data, ids=ids, winners=winners, a=a, b=b)

    def load_feature_rule(self) -> npt.NDArray[np.intp] | None:
        if self.use_cache and self.feature_rule is not None:
            return self.feature_rule
        if os.path.exists(feature_rule):
            try:
                data = np.load(feature_rule)
                if "kept_indices" in data:
                    self.feature_rule = data["kept_indices"]
                    return self.feature_rule
            except Exception:
                pass
        return None

    def save_feature_rule(self, kept_indices: npt.NDArray[np.intp]) -> None:
        logger.debug("saving feature rule...")
        os.makedirs(models_dir, exist_ok=True)
        np.savez_compressed(feature_rule, kept_indices=kept_indices)
        if self.use_cache:
            self.feature_rule = kept_indices

    def load_comparison_rule(
        self, threshold: int
    ) -> dict[str, tuple[float, int]] | None:
        """Return the cached subset rule for this threshold.

        The rule maps kept filename -> (score, count), where the score is
        recomputed on the kept subset. It is only valid for the threshold it was
        built with, so a mismatch is treated as a cache miss.
        """
        if self.use_cache and self.comparison_rule is not None:
            cached_threshold, rule = self.comparison_rule
            if cached_threshold == threshold:
                return rule
        if os.path.exists(comparison_rule):
            try:
                data = np.load(comparison_rule, allow_pickle=True)
                if all(k in data for k in ("threshold", "keys", "scores", "counts")):
                    if int(data["threshold"]) != threshold:
                        return None
                    keys = [str(f) for f in data["keys"]]
                    score_array = data["scores"]
                    count_array = data["counts"]
                    rule = {
                        keys[i]: (float(score_array[i]), int(count_array[i]))
                        for i in range(len(keys))
                    }
                    if self.use_cache:
                        self.comparison_rule = (threshold, rule)
                    return rule
            except Exception:
                pass
        return None

    def save_comparison_rule(
        self, threshold: int, rule: dict[str, tuple[float, int]]
    ) -> None:
        logger.debug("saving comparison rule...")
        os.makedirs(models_dir, exist_ok=True)
        keys = sorted(rule)
        keys_array = np.array(keys, dtype=object)
        scores_array = np.array([rule[k][0] for k in keys], dtype=np.float32)
        counts_array = np.array([rule[k][1] for k in keys], dtype=np.int64)
        np.savez_compressed(
            comparison_rule,
            threshold=np.int64(threshold),
            keys=keys_array,
            scores=scores_array,
            counts=counts_array,
        )
        if self.use_cache:
            self.comparison_rule = (threshold, rule)

    def load_interaction_data(
        self,
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.intp]] | None:
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
        self, x: npt.NDArray[np.float32], top_k_indices_local: npt.NDArray[np.intp]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.intp]]:
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
