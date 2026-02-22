import numpy as np
import lightgbm as lgb
from tqdm import tqdm
import gc
from typing import Any, Tuple, Dict, List
from sklearn.preprocessing import PolynomialFeatures

from ..config import config
from ..loaders.training_loader import training_loader


class DataTransformer:
    def __init__(self) -> None:
        self.poly = PolynomialFeatures(
            degree=2, include_bias=False, interaction_only=True
        )
        pass

    def filter_unused_features(
        self, x: np.ndarray, y: np.ndarray, steps: int, verbose: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Trains a fast LightGBM model to identify and remove features with zero importance.
        Returns the filtered X dataset and the indices of the kept features.
        Handles its own caching via 'filtered_data_cache.npz'.
        """
        filtered_data_cached = training_loader.load_filtered_data()
        if filtered_data_cached:
            return filtered_data_cached

        device_name: str = str(config["training"]["device"])
        user_verbosity = int(config["training"]["verbosity"])
        if device_name == "cuda":
            device_name = "gpu"

        params: Dict[str, Any] = {
            "objective": "regression",
            "metric": "l2",
            "verbosity": -1,
            "n_estimators": steps,  # Enough to find gradients
            "learning_rate": 0.1,
            "min_child_samples": 1,
            "device_type": device_name,
        }
        model = lgb.LGBMRegressor(**params)

        # Setup callbacks for logging
        callbacks: List[Any] = []
        # Always suppress default logger to ensure clean output
        callbacks.append(lgb.log_evaluation(period=-1))

        if user_verbosity >= 0:
            # Use tqdm progress bar
            pbar = tqdm(total=params["n_estimators"], desc="Training LightGBM")

            def pbar_callback(_):
                pbar.update(1)

            callbacks.append(pbar_callback)

        model.fit(x, y, callbacks=callbacks)

        importances = model.feature_importances_
        mask = importances > 0
        kept_indices = np.where(mask)[0]

        if len(kept_indices) == 0:
            if verbose:
                print("Warning: No important features found. Keeping all features.")
            return (x, np.arange(x.shape[1]))

        X_filtered = x[:, mask]

        if verbose:
            n_dropped = x.shape[1] - X_filtered.shape[1]
            print(f"Dropped {n_dropped} unused features. New shape: {X_filtered.shape}")

        filtered_data = training_loader.save_filtered_data(X_filtered, kept_indices)
        if verbose:
            print(f"Saved filtered data to cache")

        return filtered_data

    def calculate_interaction_batch(
        self,
        X_batch: np.ndarray,
        y_batch: np.ndarray,
        n_features_in: int,
        accumulators: Dict[str, Any],
    ) -> Dict[str, Any]:

        # Generate Poly
        X_poly_full: np.ndarray = self.poly.fit_transform(X_batch)
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
        self, k: int, accumulators: Dict[str, Any], n_samples: int, dtype
    ) -> Tuple[np.ndarray, np.ndarray]:
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
        X_interactions = np.zeros((n_samples, k), dtype=dtype)
        return X_interactions, top_k_indices_local

    def build_interaction_batch(
        self, X_batch: np.ndarray, top_k_indices_local: np.ndarray, n_features_in: int
    ) -> np.ndarray:
        X_poly_full = self.poly.fit_transform(X_batch)
        current_interactions = X_poly_full[:, n_features_in:][:, top_k_indices_local]
        del X_poly_full
        gc.collect()
        return current_interactions

    def add_interaction_features(
        self,
        x: np.ndarray,
        y: np.ndarray,
        target_k: int = 500,
    ) -> Tuple[np.ndarray, np.ndarray]:
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
        BATCH_MEMORY_TARGET = 1 * 1024**3
        # Approx full poly width for memory calc
        n_features_total = n_features_in + n_interactions
        bytes_per_row = n_features_total * 8
        batch_size = int(BATCH_MEMORY_TARGET / bytes_per_row)
        if batch_size < 1:
            batch_size = 1
        if batch_size > 5000:
            batch_size = 5000

        # Accumulators for correlation calculation

        accumulators: Dict[str, Any] = {
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

    def apply_feature_filter(self, vecs: List[np.ndarray]) -> List[np.ndarray]:
        """
        Applies the feature filter (kept_indices) from filtered_data_cache.npz to the input vector.
        model_bin_dir: directory containing filtered_data_cache.npz
        """

        filtered_data_cached = training_loader.load_filtered_data()
        if not filtered_data_cached:
            raise FileNotFoundError("Training data not found, must generate first")

        _, kept_indices = filtered_data_cached
        results: List[np.ndarray] = []
        for vec in vecs:
            filtered_vector = vec[kept_indices]
            results.append(filtered_vector)
        return results

    def apply_interaction_features(self, vecs: List[np.ndarray]) -> np.ndarray:
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
