import numpy as np
from numpy import typing as npt
import lightgbm as lgb
from tqdm import tqdm
import gc
from typing import Any
from sklearn.preprocessing import PolynomialFeatures

from ..config import config
from ..loaders.training_loader import training_loader
from .model_trainer import model_trainer


class DataTransformer:

    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)

    def __init__(self) -> None:
        pass

    def filter_unused_features(
        self,
        x: npt.NDArray[np.float32],
        y: npt.NDArray[np.float32],
        steps: int,
        verbose: bool = True,
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """
        Trains a fast LightGBM model to identify and remove features with zero importance
        and low cumulative gain. Returns the filtered X dataset and the indices of kept features.
        Handles its own caching via 'filtered_data_cache.npz'.
        """
        filtered_data_cached = training_loader.load_filtered_data()
        if filtered_data_cached:
            return filtered_data_cached

        user_verbosity = int(config["training"]["verbosity"])

        # Create training model using shared trainer with minimal config
        config_dict: dict[str, Any] = {
            "n_estimators": steps,
        }
        model_trainer.create_training_model(config_dict)
        model: None | lgb.LGBMRanker | lgb.LGBMClassifier | lgb.LGBMRegressor = (
            model_trainer.training_model
        )

        # Setup callbacks for logging
        callbacks: list[Any] = [
            lgb.log_evaluation(period=-1)
        ]  # suppress default logger

        if user_verbosity >= 0:
            # Use tqdm progress bar
            pbar = tqdm(total=steps, desc="Training LightGBM")

            def pbar_callback(_):
                pbar.update(1)

            callbacks.append(pbar_callback)

        # Train the model
        model.fit(x, y, callbacks=callbacks)

        # Get feature importances (gain)
        importances: np.ndarray[tuple[Any, ...], np.dtype[Any]] = (
            model.feature_importances_
        )
        n_features = len(importances)

        if verbose:
            n_zeros = np.sum(importances == 0)
            print(
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
        cum_threshold = 0.95
        keep_mask = cumulative <= cum_threshold

        # Always keep at least one feature
        if not np.any(keep_mask):
            keep_mask[0] = True

        kept_indices = sorted_indices[keep_mask]

        # Apply mask to X
        X_filtered = x[:, kept_indices]

        if verbose:
            n_dropped = x.shape[1] - X_filtered.shape[1]
            print(f"Dropped {n_dropped} features. New shape: {X_filtered.shape}")

        # Cache filtered data
        filtered_data = training_loader.save_filtered_data(X_filtered, kept_indices)
        if verbose:
            print("Saved filtered data to cache")

        return filtered_data

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
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
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
        top_k_indices_local: npt.NDArray[np.float32],
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
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
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

        print(
            f"Scanning {n_interactions} potential interactions in batches of {batch_size}..."
        )

        # Pass 1: Stats
        with tqdm(
            total=n_samples, desc="Computing Correlations", unit="samples"
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
            total=n_samples, desc="Building Interaction Matrix", unit="samples"
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
        print(f"Saved interaction data to cache")

        return interaction_data

    def apply_feature_filter(
        self, vecs: list[npt.NDArray[np.float32]]
    ) -> list[npt.NDArray[np.float32]]:
        """
        Applies the feature filter (kept_indices) from filtered_data_cache.npz to the input vector.
        model_bin_dir: directory containing filtered_data_cache.npz
        """

        filtered_data_cached = training_loader.load_filtered_data()
        if not filtered_data_cached:
            raise FileNotFoundError("Training data not found, must generate first")

        _, kept_indices = filtered_data_cached
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


data_transformer = DataTransformer()
