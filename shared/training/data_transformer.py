import os
import json
import numpy as np
from numpy import typing as npt
import lightgbm as lgb
from tqdm import tqdm
import gc
from typing import Any
from sklearn.preprocessing import PolynomialFeatures

from ..logger import get_logger
from ..config import config
from ..loaders.training_loader import training_loader
from .model_trainer import model_trainer
from ...external_modules.comparison.algorithm.trueskill_rating import (
    public_score_from_rating,
    replay_ratings,
)

logger = get_logger(__name__)


def get_feature_mapping_from_config() -> dict[str, Any]:
    """
    Creates a mapping from feature indices to vector names and positions.
    Returns both a forward map (index -> vector info) and reverse map (vector -> indices).
    """
    feature_to_vector = {}
    vector_ranges = {}
    current_idx = 0
    config_vectors: list[dict[str, Any]] = config["vector"]["vectors"]
    for vector_config in config_vectors:
        vec_name = vector_config["name"]
        slot_size = vector_config["slot_size"]
        entry: dict[str, Any] = {
            "start_idx": current_idx,
            "end_idx": current_idx + slot_size,
            "slot_size": slot_size,
            "type": vector_config["type"],
        }
        if "per_unit_size" in vector_config:
            entry["per_unit_size"] = vector_config["per_unit_size"]
        vector_ranges[vec_name] = entry

        for i in range(slot_size):
            feature_to_vector[current_idx + i] = {
                "vector_name": vec_name,
                "position_in_vector": i,
                "total_in_vector": slot_size,
            }

        current_idx += slot_size

    return {
        "feature_to_vector": feature_to_vector,
        "vector_ranges": vector_ranges,
        "total_features": current_idx,
    }


class DataTransformer:
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)

    def __init__(self) -> None:
        feature_mapping = get_feature_mapping_from_config()
        self.feature_to_vector: dict[int, dict[str, Any]] = feature_mapping[
            "feature_to_vector"
        ]
        self.vector_ranges = feature_mapping["vector_ranges"]

    def get_raw_data(self) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        vectors = training_loader.load_vectors()
        scores = training_loader.load_scores()
        order = list(scores.keys())
        x = np.array([vectors[fid] for fid in order], dtype=np.float32)
        y = np.array([scores[fid] for fid in order], dtype=np.float32)
        return x, y

    def filter_low_comparisons(
        self,
        threshold: int = 0,
    ) -> dict[str, tuple[float, int]]:
        """Return the kept subset as filename -> (score, count).

        Filenames with at least ``threshold`` comparisons are kept, then the
        TrueSkill ratings are replayed using only comparisons between kept
        filenames so the surviving scores are self-consistent for that subset.
        The result is cached in comparison_rule.npz keyed by threshold.
        """
        rule_cached = training_loader.load_comparison_rule(threshold)
        if rule_cached is not None:
            return rule_cached

        rows = training_loader.load_comparison_rows()

        counts: dict[str, int] = {}
        for row in rows:
            counts[row["filename_a"]] = counts.get(row["filename_a"], 0) + 1
            counts[row["filename_b"]] = counts.get(row["filename_b"], 0) + 1
        kept_filenames = {fid for fid, count in counts.items() if count >= threshold}

        subset_rows = [
            row
            for row in rows
            if row["filename_a"] in kept_filenames
            and row["filename_b"] in kept_filenames
        ]
        replayed = replay_ratings(subset_rows)
        rule = {
            fid: (public_score_from_rating(rating), count)
            for fid, (rating, count) in replayed.items()
        }

        training_loader.save_comparison_rule(threshold, rule)
        return rule

    def filter_unused_features(
        self,
        vectors_keyed: dict[str, npt.NDArray[np.float32]],
        scores_keyed: dict[str, float],
        steps: int,
        verbose: bool = True,
    ) -> npt.NDArray[np.intp]:
        """
        Trains a fast LightGBM model on the keyed vectors/scores to identify and
        remove features with zero importance and low cumulative gain. Returns
        only the kept column indices (the feature mask). Caches the mask as
        feature_rule.npz. No sorting or alignment is performed here; the keyed
        inputs are already consistent.
        """

        rule_cached = training_loader.load_feature_rule()
        if rule_cached is not None:
            return rule_cached

        # Build (x, y) from the keyed inputs without imposing any ordering.
        order = list(scores_keyed.keys())
        x = np.array([vectors_keyed[fid] for fid in order], dtype=np.float32)
        y = np.array([scores_keyed[fid] for fid in order], dtype=np.float32)

        user_verbosity = int(config["training"]["verbosity"])

        # Create training model using shared trainer with minimal config
        config_dict: dict[str, Any] = {
            "n_estimators": steps,
        }
        if config["training"]["objective"] == "lambdarank":
            # Feature filtering still uses the scalar score target even when
            # the final training objective is pairwise ranking.
            config_dict["objective"] = "regression"
        model_trainer.create_training_model(config_dict)
        model = model_trainer.training_model
        assert model is not None

        # Setup callbacks for logging
        callbacks: list[Any] = [
            lgb.log_evaluation(period=-1)
        ]  # suppress default logger
        pbar = None
        if user_verbosity >= 0:
            # Use tqdm progress bar
            pbar = tqdm(total=steps, desc="Training LightGBM", delay=3.0)

            def pbar_callback(_):
                pbar.update(1)

            callbacks.append(pbar_callback)

        # Train the model
        model.fit(x, y, callbacks=callbacks)
        if pbar is not None:
            pbar.close()

        # Get feature importances (gain)
        importances: np.ndarray[tuple[Any, ...], np.dtype[Any]] = (
            model.feature_importances_
        )
        n_features = len(importances)

        n_zeros = np.sum(importances == 0)
        logger.debug(
            f"Found {n_zeros} features with zero gain out of {n_features} total features."
        )

        # --- Step 1: remove all zero-gain features
        nonzero_mask = importances > 0
        nonzero_importances = importances[nonzero_mask]
        nonzero_indices = np.where(nonzero_mask)[0]

        # --- Step 2: cumulative gain pruning
        sorted_idx = np.argsort(nonzero_importances)[::-1]  # descending
        sorted_importances = nonzero_importances[sorted_idx]
        sorted_indices = nonzero_indices[sorted_idx]

        cumulative = np.cumsum(sorted_importances)
        cumulative /= cumulative[-1]  # normalize to 1

        # Keep features until cumulative gain reaches threshold (e.g., 95%)
        cum_threshold = 0.99
        keep_mask = cumulative <= cum_threshold

        # Always keep at least one feature
        if not np.any(keep_mask):
            keep_mask[0] = True

        kept_indices = sorted_indices[keep_mask]

        # Cache only the mask (kept column indices); the caller applies it.
        training_loader.save_feature_rule(kept_indices)
        logger.debug("Saved feature rule to cache")

        return kept_indices

    def calculate_interaction_batch(
        self,
        X_batch: npt.NDArray[np.float32],
        y_batch: npt.NDArray[np.float32],
        n_features_in: int,
        accumulators: dict[str, Any],
    ) -> dict[str, Any]:

        # Generate Poly
        X_poly_full: npt.NDArray[np.float32] = self.poly.fit_transform(X_batch)
        # Extract only interactions
        X_inter_batch = X_poly_full[:, n_features_in:]

        # Stats
        accumulators["sum_x"] += np.sum(X_inter_batch, axis=0)
        accumulators["sum_x_sq"] += np.sum(X_inter_batch**2, axis=0)
        accumulators["sum_xy"] += np.dot(X_inter_batch.T, y_batch)

        accumulators["sum_y"] += np.sum(y_batch)
        accumulators["sum_y_sq"] += np.sum(y_batch**2)
        accumulators["n"] += len(X_batch)

        del X_poly_full, X_inter_batch
        gc.collect()
        return accumulators

    def compute_correlations(
        self, k: int, accumulators: dict[str, Any], n_samples: int, dtype: np.dtype[Any]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.intp]]:
        n = accumulators["n"]
        # Compute Correlations (Pearson)
        numerator = (n * accumulators["sum_xy"]) - (
            accumulators["sum_x"] * accumulators["sum_y"]
        )
        denominator_x = (n * accumulators["sum_x_sq"]) - (accumulators["sum_x"] ** 2)
        denominator_y = (n * accumulators["sum_y_sq"]) - (accumulators["sum_y"] ** 2)

        # Avoid div by zero / invalid sqrt
        denominator_x[denominator_x <= 0] = 1e-10
        denominator = np.sqrt(denominator_x * denominator_y)

        correlation = numerator / denominator
        f_scores = (correlation**2) / (1 - correlation**2 + 1e-10) * (n - 2)
        f_scores = np.nan_to_num(f_scores, nan=0.0)

        top_k_indices_local = np.argsort(f_scores)[-k:]
        top_k_indices_local = np.sort(top_k_indices_local)
        # Pass 2: Build
        X_interactions: npt.NDArray[np.float32] = np.zeros((n_samples, k), dtype=dtype)
        return X_interactions, top_k_indices_local

    def build_interaction_batch(
        self,
        X_batch: npt.NDArray[np.float32],
        top_k_indices_local: npt.NDArray[np.intp],
        n_features_in: int,
    ) -> npt.NDArray[np.float32]:
        X_poly_full = self.poly.fit_transform(X_batch)
        current_interactions = X_poly_full[:, n_features_in:][:, top_k_indices_local]
        del X_poly_full
        gc.collect()
        return current_interactions

    def add_interaction_features(
        self,
        x: npt.NDArray[np.float32],
        y: npt.NDArray[np.float32],
        target_k: int = 500,
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.intp]]:
        """
        Generates and selects top K interaction features (x*y) using batched processing
        to avoid OOM. Concatenates them to X.
        Returns (X_final, selected_interaction_indices)
        Handles its own caching via 'interaction_data_cache.npz'.
        """

        interaction_data_cached = training_loader.load_interaction_data()
        if interaction_data_cached:
            return interaction_data_cached

        n_features_in = x.shape[1]
        n_samples = x.shape[0]

        # Basic sizing check before proceeding
        # n*(n-1)/2 interactions

        # We only care about columns [n_features_in : ] from the poly expansion
        # Calculate number of interactions
        n_interactions = (n_features_in * (n_features_in - 1)) // 2

        if n_interactions == 0:
            return x, np.array([])

        # Batch Size Calculation (Target 1GB)
        BATCH_MEMORY_TARGET = 8 * 1024**3
        # Approx full poly width for memory calc
        n_features_total = n_features_in + n_interactions
        bytes_per_row = n_features_total * 8
        batch_size = int(BATCH_MEMORY_TARGET / bytes_per_row)
        if batch_size < 1:
            batch_size = 1
        if batch_size > 5000:
            batch_size = 5000

        # Accumulators for correlation calculation

        accumulators: dict[str, Any] = {
            "sum_x": np.zeros(n_interactions),
            "sum_x_sq": np.zeros(n_interactions),
            "sum_xy": np.zeros(n_interactions),
            "sum_y": 0,
            "sum_y_sq": 0,
            "n": 0,
        }

        logger.debug(
            f"Scanning {n_interactions} potential interactions in batches of {batch_size}..."
        )

        # Pass 1: Stats
        with tqdm(
            total=n_samples, desc="Computing Correlations", unit="samples", delay=3.0
        ) as pbar:
            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)
                X_batch = x[i:end_idx]
                y_batch = y[i:end_idx]
                accumulators = self.calculate_interaction_batch(
                    X_batch, y_batch, n_features_in, accumulators
                )
                pbar.update(len(X_batch))

        # Select Top K
        k = min(target_k, n_interactions)

        x_interactions, top_k_indices_local = self.compute_correlations(
            k, accumulators, n_samples, x.dtype
        )

        with tqdm(
            total=n_samples, desc="Building Interaction Matrix", unit="samples", delay=3.0
        ) as pbar:
            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)
                X_batch = x[i:end_idx]
                current_interactions = self.build_interaction_batch(
                    X_batch, top_k_indices_local, n_features_in
                )

                x_interactions[i:end_idx] = current_interactions

                pbar.update(len(X_batch))

        X_final = np.hstack([x, x_interactions])

        # --- Save Cache ---
        interaction_data = training_loader.save_interaction_data(
            X_final, top_k_indices_local
        )
        logger.debug(f"Saved interaction data to cache")

        return interaction_data

    def apply_feature_filter(
        self, vecs: list[npt.NDArray[np.float32]]
    ) -> list[npt.NDArray[np.float32]]:
        """
        Applies the feature filter (kept_indices) from feature_rule.npz to the input vector.
        """

        rule = training_loader.load_feature_rule()
        if rule is None:
            raise FileNotFoundError("Feature rule not found, must generate first")

        kept_indices = rule
        results: list[npt.NDArray[np.float32]] = []
        for vec in vecs:
            filtered_vector = vec[kept_indices]
            results.append(filtered_vector)
        return results

    def apply_interaction_features(
        self, vecs: list[npt.NDArray[np.float32]]
    ) -> npt.NDArray[np.float32]:
        """
        Applies the interaction features (from interaction_data_cache.npz) to the input vector.
        model_bin_dir: directory containing interaction_data_cache.npz
        """
        interaction_data_cached = training_loader.load_interaction_data()
        if not interaction_data_cached:
            raise FileNotFoundError("Interaction data not found, must generate first")

        vecs_np = np.array(vecs)  # shape: (batch_size, feature_dim)
        poly_features = self.poly.fit_transform(
            vecs_np
        )  # shape: (batch_size, n_poly_features)

        n_features_in = vecs_np.shape[1]
        _, interaction_indices = interaction_data_cached
        inter_feats = poly_features[:, n_features_in:][:, interaction_indices]

        # Concatenate original features and selected interaction features
        X_final = np.hstack([vecs_np, inter_feats])
        return X_final


def _label_position_slot(pos_in_unit: int) -> str:
    return (
        ["x", "y", "width", "height", "confidence"][pos_in_unit]
        if pos_in_unit < 5
        else f"slot_{pos_in_unit}"
    )


def _label_keypoint_slot(vec_name: str, pos_in_unit: int) -> str:
    coord = (
        ["x", "y", "z", "visibility"][pos_in_unit]
        if pos_in_unit < 4
        else f"slot_{pos_in_unit}"
    )
    return f"{vec_name}_{coord}"


def _label_person_map_slot(vec_name: str, pos_in_unit: int) -> str:
    labels = _load_map_slots(vec_name)
    if labels and pos_in_unit < len(labels):
        return labels[pos_in_unit]
    return f"slot_{pos_in_unit}"


def _load_map_slots(vec_name: str) -> list[str] | None:
    """Load the saved map JSON for a map-type vector, returning slot labels by index."""
    from ..paths import maps_dir

    path = os.path.join(maps_dir, f"{vec_name}_map.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return None


def _print_vector_summary(
    vec_name: str,
    vec_type: str,
    kept_in_vec: list[int],
    total_in_vec: int,
    slot_size: int,
    per_unit_size: int | None,
    start_idx: int = 0,
) -> None:
    """Print one vector line (or expanded lines for known multi-slot / map vectors)."""
    kept_count = len(kept_in_vec)
    pct = 100.0 * kept_count / total_in_vec if total_in_vec else 0

    if per_unit_size and per_unit_size > 1:
        # Multi-slot vector (position / keypoint / person_map) broken down per unit
        n_units = slot_size // per_unit_size
        local_kept = [i - start_idx for i in kept_in_vec]
        unit_labels_fn = None
        if vec_name == "bbox":
            unit_labels_fn = lambda upos: _label_position_slot(upos)
        elif vec_type == "keypoint":
            unit_labels_fn = lambda upos: _label_keypoint_slot(vec_name, upos)
        elif vec_type == "person_map":
            unit_labels_fn = lambda upos: _label_person_map_slot(vec_name, upos)

        print(f"  {vec_name}  ({kept_count}/{total_in_vec} = {pct:.1f}%)")
        if unit_labels_fn:
            for ui in range(n_units):
                offset = ui * per_unit_size
                kept_in_unit = [
                    i for i in local_kept if offset <= i < offset + per_unit_size
                ]
                if kept_in_unit:
                    kept_positions = [i - offset for i in kept_in_unit]
                    labels = [unit_labels_fn(p) for p in kept_positions]
                    print(f"    unit {ui}: kept({', '.join(labels)})")
                elif n_units > 1:
                    pass  # skip units with nothing kept
            if kept_count == 0 and n_units == 1 and unit_labels_fn:
                all_labels = [unit_labels_fn(p) for p in range(per_unit_size)]
                print(f"    Sub-features: {', '.join(all_labels)}")
        return

    # Map vector — convert absolute indices to local slots and look up labels
    if vec_type == "map" and kept_count > 0:
        slot_labels = _load_map_slots(vec_name)
        print(f"  {vec_name}  ({kept_count}/{total_in_vec} = {pct:.1f}%)")
        for i in kept_in_vec:
            local_slot = i - start_idx
            label = (
                slot_labels[local_slot]
                if slot_labels and local_slot < len(slot_labels)
                else f"slot_{local_slot}"
            )
            print(f"    [{local_slot}] {label}")
        return

    # Simple line per vector
    print(f"  {vec_name:28s}  {kept_count:4d}/{total_in_vec:4d}  ({pct:5.1f}%)")


def list_filtered_features() -> None:
    """
    Loads the cached filtered features and prints a compact summary of which
    features survived the gain-based pruning. Multi-slot vectors (position,
    keypoint, person_map) are broken down by named sub-feature.
    """
    cached = training_loader.load_feature_rule()
    if cached is None:
        print(
            "No feature rule cache found. Run data_transformer.filter_unused_features() first."
        )
        return

    kept_indices = cached
    mapping = get_feature_mapping_from_config()
    total = mapping["total_features"]
    kept_set = set(kept_indices.tolist())

    print(f"Total features: {total}")
    print(f"Kept: {len(kept_indices)}, Dropped: {total - len(kept_indices)}")
    print()

    # Group kept indices by vector
    vec_kept: dict[str, list[int]] = {vn: [] for vn in mapping["vector_ranges"]}
    for i in sorted(kept_set):
        info = mapping["feature_to_vector"].get(i)
        if info:
            vec_kept[info["vector_name"]].append(i)

    print("--- Feature survival summary ---")
    for vec_name in sorted(mapping["vector_ranges"].keys()):
        rng = mapping["vector_ranges"][vec_name]
        slot_size = rng["slot_size"]
        total_in_vec = rng["end_idx"] - rng["start_idx"]
        kept_in_vec = vec_kept.get(vec_name, [])

        _print_vector_summary(
            vec_name,
            rng.get("type", ""),
            kept_in_vec,
            total_in_vec,
            slot_size,
            rng.get("per_unit_size"),
            start_idx=rng["start_idx"],
        )


data_transformer = DataTransformer()
