"""Data Transformer Module - transforms data for model training."""
import numpy.typing as npt
import numpy as np
from typing import Any
from tqdm import tqdm

from ..loaders.training_loader import training_loader


class DataTransformer:
    """Transforms training data with interaction features."""

    def __init__(self) -> None:
        self.use_cache = True
        self.interaction_data_cache: tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]] | None = None

    def add_interaction_features(
        self,
        x: npt.NDArray[np.float32],
        y: npt.NDArray[np.float32],
        target_k: int = 500,
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        from ..loaders.training_loader import training_loader

        interaction_data_cached = training_loader.load_interaction_data()  # type: ignore[attr-defined]
        if interaction_data_cached:
            return interaction_data_cached

        n_features_in = x.shape[1]
        n_samples = x.shape[0]

        n_interactions = (n_features_in * (n_features_in - 1)) // 2

        if n_interactions == 0:
            return x, np.array([])

        target_k = min(target_k, n_interactions)

        accumulators = {
            "sum_x": np.zeros(n_interactions),
            "sum_x_sq": np.zeros(n_interactions),
            "sum_xy": np.zeros(n_interactions),
            "sum_y": 0.0,
            "sum_y_sq": 0.0,
            "n": 0,
        }

        batch_size = min(5000, max(1, n_samples))

        print(
            f"Scanning {n_interactions} potential interactions in batches of {batch_size}..."
        )

        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            x_batch = x[i:end_idx]

            x_poly = self._polynomial_features(x_batch, n_features_in)
            interactions = x_poly[:, n_features_in:]

            accumulators["sum_x"] += interactions.sum(axis=0)
            accumulators["sum_x_sq"] += (interactions ** 2).sum(axis=0)
            accumulators["sum_xy"] += (interactions * y[i:end_idx, np.newaxis]).sum(axis=0)
            accumulators["sum_y"] += y[i:end_idx].sum()
            accumulators["sum_y_sq"] += (y[i:end_idx] ** 2).sum()
            accumulators["n"] += len(y[i:end_idx])

        correlations = self._compute_correlations(target_k, accumulators, n_samples)  # type: ignore[attr-defined]

        x_interactions = self._build_interactions(x, correlations[1], n_features_in)  # type: ignore[attr-defined]

        X_final = np.hstack([x, x_interactions])

        interaction_data = training_loader.save_interaction_data(  # type: ignore[attr-defined]
            X_final, correlations[1]  # type: ignore[arg-type]
        )
        print(f"Saved interaction data to cache")

        return interaction_data

    def _polynomial_features(
        self, x: npt.NDArray[np.float32], n_features: int
    ) -> npt.NDArray[np.float32]:
        from sklearn.preprocessing import PolynomialFeatures

        poly = PolynomialFeatures(degree=2, include_bias=False)
        return poly.fit_transform(x)

    def _compute_correlations(
        self, k: int, accumulators: dict[str, Any], n_samples: int
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        n = accumulators["n"]
        numerator = (n * accumulators["sum_xy"]) - (accumulators["sum_x"] * accumulators["sum_y"])
        denominator_x = (n * accumulators["sum_x_sq"]) - (accumulators["sum_x"] ** 2)
        denominator_y = (n * accumulators["sum_y_sq"]) - (accumulators["sum_y"] ** 2)

        denominator_x[denominator_x <= 0] = 1e-10
        denominator = np.sqrt(denominator_x * denominator_y)

        correlation = numerator / denominator
        f_scores = (correlation**2) / (1 - correlation**2 + 1e-10) * (n - 2)
        f_scores = np.nan_to_num(f_scores, nan=0.0)

        top_k_indices = np.argsort(f_scores)[-k:]
        top_k_indices = np.sort(top_k_indices)

        X_interactions: npt.NDArray[np.float32] = np.zeros((n_samples, k), dtype=np.float32)
        return X_interactions, top_k_indices  # type: ignore[return-value]

    def _build_interactions(
        self, x: npt.NDArray[np.float32], top_k_indices: npt.NDArray[np.float32], n_features: int
    ) -> npt.NDArray[np.float32]:
        x_poly = self._polynomial_features(x, n_features)
        interactions = x_poly[:, n_features:]
        return interactions[:, top_k_indices]  # type: ignore[index]

    def apply_feature_filter(
        self, vecs: list[npt.NDArray[np.float32]]
    ) -> list[npt.NDArray[np.float32]]:
        filtered_data_cached = training_loader.load_filtered_data()  # type: ignore[attr-defined]
        if not filtered_data_cached:
            raise FileNotFoundError("Training data not found, must generate first")

        _, kept_indices = filtered_data_cached
        results: list[npt.NDArray[np.float32]] = []
        for vec in vecs:
            kept_indices_int = kept_indices.astype(np.intp)  # type: ignore[attr-defined]
            filtered_vector = vec[kept_indices_int]  # type: ignore[index]
            results.append(filtered_vector)
        return results


# Global instance
data_transformer = DataTransformer()