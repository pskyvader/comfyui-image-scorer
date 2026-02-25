from typing import Any, Dict, List, Tuple, Optional
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

from ..loaders.training_loader import training_loader
from ..config import config


class PlotManager:
    """Manages all plotting functionality for model training and analysis."""

    # Metric direction mapping (shared across methods)
    METRIC_DIRECTIONS: Dict[str, Dict[str, bool]] = {
        "lambdarank": {"ndcg": True},
        "multiclass": {"multi_logloss": False, "multi_error": False},
        "binary": {"binary_logloss": False, "auc": True},
        "regression": {"l2": False, "rmse": False, "l1": False, "r2": True},
    }

    @staticmethod
    def _get_metric_direction(objective: str, metric: str) -> bool:
        """Get whether higher is better for a given metric."""
        return PlotManager.METRIC_DIRECTIONS[objective].get(metric, True)

    @staticmethod
    def _prepare_finite_data(
        y: Any, preds: Any
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Filter out non-finite values from y and predictions."""
        y_sample = np.asarray(y).ravel()
        preds = np.asarray(preds).ravel()
        mask = np.isfinite(y_sample) & np.isfinite(preds)
        if not mask.any():
            return np.array([]), np.array([])
        return y_sample[mask], preds[mask]

    @staticmethod
    def _calculate_scatter_sizes(
        counts: np.ndarray,
        min_size_px: float = 1.0,
        max_size_px: float = 800.0,
        power: float = 0.5,
    ) -> np.ndarray:
        """Calculate scatter plot marker sizes based on counts."""
        counts_scaled = counts**power
        counts_norm = (counts_scaled - counts_scaled.min()) / (
            counts_scaled.max() - counts_scaled.min() + 1e-12
        )
        return min_size_px + counts_norm * (max_size_px - min_size_px)

    @staticmethod
    def _setup_scatter_axes(ax: Any, y_min: float, y_max: float) -> None:
        """Configure scatter plot axes and add diagonal reference line."""
        if np.isfinite(y_min) and np.isfinite(y_max):
            margin = max((y_max - y_min) * 0.02, 1e-6)
            ax.plot(
                [y_min - margin, y_max + margin],
                [y_min - margin, y_max + margin],
                "r--",
                label="perfect prediction",
                linewidth=2.0,
                zorder=10,
            )
            ax.set_xlim(y_min - margin, y_max + margin)
            ax.set_ylim(y_min - margin, y_max + margin)
            ax.set_aspect("equal", adjustable="box")

    @staticmethod
    def plot_scatter_comparison(
        y_plot: np.ndarray,
        p_plot: np.ndarray,
        plot: bool = True,
        min_size_px: float = 1.0,
        max_size_px: float = 800.0,
        power: float = 0.5,
        label_threshold: int = 10,
    ) -> None:
        """Plot actual vs predicted values with cluster sizes."""
        points = np.column_stack((y_plot, p_plot))
        unique_points, counts = np.unique(points, axis=0, return_counts=True)

        sizes = PlotManager._calculate_scatter_sizes(
            counts, min_size_px, max_size_px, power
        )

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(
            unique_points[:, 0],
            unique_points[:, 1],
            s=sizes,
            alpha=0.85,
            edgecolors="none",
            zorder=5,
        )

        # Add labels for high-count points
        for (x, y), c in zip(unique_points, counts):
            if c >= label_threshold:
                ax.annotate(
                    str(c),
                    (x, y),
                    textcoords="offset points",
                    xytext=(3, 3),
                    fontsize=8,
                    ha="left",
                    va="bottom",
                    zorder=20,
                )

        ymin = float(min(np.min(y_plot), np.min(p_plot)))
        ymax = float(max(np.max(y_plot), np.max(p_plot)))

        PlotManager._setup_scatter_axes(ax, ymin, ymax)

        ax.legend()
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Actual vs Predicted (sample)")
        ax.grid(True)

        if plot:
            plt.show()

    @staticmethod
    def prepare_plot_data(y: Any, preds: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for plotting by filtering out non-finite values."""
        y_sample, preds_sample = PlotManager._prepare_finite_data(y, preds)
        if y_sample.size == 0:
            print("No finite prediction/data pairs to plot; skipping compare plot.")
        return y_sample, preds_sample

    @staticmethod
    def print_comparison_metrics(y: Any, preds: Any, metrics: Any) -> None:
        """Print comparison metrics between predictions and true values."""
        y_sample, p_sample = PlotManager._prepare_finite_data(y, preds)
        if y_sample.size > 0:
            sample_r2 = float(r2_score(y_sample, p_sample))
            print(f"Comparison metrics (sample): r2={sample_r2:.4f}, n={len(y_sample)}")
        if metrics is not None and "r2" in metrics:
            print(f"Stored metrics: r2={float(metrics['r2']):.4f}")

    @staticmethod
    def compare_model_vs_data(
        x: np.ndarray, y: np.ndarray, plot: bool = True, limit: int = 500
    ) -> None:
        """Compare a trained model vs data."""
        rng = np.random.default_rng()
        indices = rng.choice(len(x), size=limit, replace=False)
        
        
        x_sample = x[indices]
        y_sample = y[indices]
        model = training_loader.load_training_model()

        if model is None:
            print("Warning: No trained model found. Skipping comparison.")
            return

        try:
            preds = model.predict(x_sample)
        except Exception as e:
            print(f"Error during prediction: {e}")
            return

        metrics = None
        data = training_loader.load_training_model_diagnostics()
        if data is not None and "metrics" in data:
            metrics = data["metrics"]

        PlotManager.print_comparison_metrics(y_sample, preds, metrics)
        y_plot, p_plot = PlotManager.prepare_plot_data(y_sample, preds)
        if y_plot.size > 0 and p_plot.size > 0:
            PlotManager.plot_scatter_comparison(y_plot, p_plot, plot)

    @staticmethod
    def _plot_metric_on_axes(
        ax: Any, metric_name: str, values: List[float], label: str, direction_higher: bool
    ) -> None:
        """Helper to plot a single metric on an axis."""
        ax.plot(values, label=f"{label} {metric_name}")
        if values:
            ax.annotate(
                f"{values[-1]:.4f}",
                xy=(len(values) - 1, values[-1]),
                xytext=(5, 0),
                textcoords="offset points",
            )

        text = ("higher" if direction_higher else "lower") + " is better"
        ax.set_xlabel("Iteration")
        ax.set_title(f"{label} {metric_name} Progress ({text})")
        ax.legend()
        ax.grid(True)

    @staticmethod
    def plot_metric(
        axes: List[Any],
        current_metric: Dict[str, List[float]],
        label: str = "Valid",
    ) -> None:
        """Plot individual metrics on subplots."""
        objective = config["training"]["objective"]
        for i, metric in enumerate(list(current_metric)):
            ax = axes[i]
            values = current_metric[metric]
            direction_higher = PlotManager._get_metric_direction(objective, metric)
            PlotManager._plot_metric_on_axes(ax, metric, values, label, direction_higher)

    @staticmethod
    def plot_loss_curve(result_metrics: Dict[str, Any] | None = None) -> None:
        """Plot loss curves from training results."""
        try:
            curves = None

            if (
                result_metrics is not None
                and "curves" in result_metrics
                and result_metrics["curves"] is not None
            ):
                curves = result_metrics["curves"]

            if curves is None:
                data = training_loader.load_training_model_diagnostics()
                if data is not None and "curves" in data:
                    curves = data["curves"]

            if curves is None:
                print("No curves available to plot.")
                return

            plt.figure(figsize=(6, 4))
            plotted = False

            for dataset_name, metrics_dict in curves.items():
                if dataset_name != "valid":
                    continue
                for metric_name, values in metrics_dict.items():
                    values = np.asarray(values, dtype=float).ravel()
                    if values.size == 0:
                        continue

                    plt.plot(
                        np.arange(1, values.size + 1),
                        values,
                        label=f"{dataset_name}_{metric_name}",
                        linewidth=1,
                    )
                    plotted = True

            if not plotted:
                print("Curves are empty; nothing to plot.")
                return

            plt.xlabel("Iteration")
            plt.ylabel("Metric Value")
            plt.title("Training Curves")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print("Failed to plot curves:", e)


class LivePlotCallback:
    """Callback to plot training progress only at the end of training."""

    def __init__(self, save_path: Optional[str] = None, frequency: int = 30, status_bar=None) -> None:
        """Initialize callback to plot only final results."""
        self.save_path = save_path
        if self.save_path:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        self.last_plot_time = time.time()
        self.history: Dict[str, Dict[str, List[float]]] = {
            "valid": {},
            "train": {},
        }
        self.frequency = frequency
        self.status_bar = status_bar

    def __call__(self, env: Any) -> None:
        """Collect metrics during training (do not plot live)."""
        for data_name, eval_name, result, _ in env.evaluation_result_list:
            if data_name not in self.history:
                self.history[data_name] = {}
            if eval_name not in self.history[data_name]:
                self.history[data_name][eval_name] = []
            self.history[data_name][eval_name].append(result)
        time_now = time.time()
        if time_now - self.last_plot_time > self.frequency:
            with self.status_bar:
                self.status_bar.set_description("Plotting final results...")
                self.plot_final_results()
                self.last_plot_time = time_now

    def plot_final_results(self) -> None:
        """Plot the final training results once, after training ends."""
        try:
            label = "valid"
            current_metrics = self.history[label]
            num_current_metrics = len(current_metrics)
            if num_current_metrics == 0:
                return

            _, axes = plt.subplots(
                1, num_current_metrics, figsize=(5 * num_current_metrics, 5)
            )
            if num_current_metrics == 1:
                axes = [axes]
            PlotManager.plot_metric(axes, current_metrics, label=label)

            plt.suptitle("Final Training Results")
            plt.tight_layout()

            if self.save_path:
                try:
                    plt.savefig(self.save_path)
                except Exception:
                    pass

            plt.show()
        except Exception as e:
            print(f"Failed to plot final results: {e}")


# Backward compatibility: function-level access
def plot_scatter_comparison(*args: Any, **kwargs: Any) -> None:
    """Backward compatible wrapper for scatter comparison plotting."""
    PlotManager.plot_scatter_comparison(*args, **kwargs)


def prepare_plot_data(*args: Any, **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Backward compatible wrapper for plot data preparation."""
    return PlotManager.prepare_plot_data(*args, **kwargs)


def print_comparison_metrics(*args: Any, **kwargs: Any) -> None:
    """Backward compatible wrapper for comparison metrics."""
    PlotManager.print_comparison_metrics(*args, **kwargs)


def compare_model_vs_data(*args: Any, **kwargs: Any) -> None:
    """Backward compatible wrapper for model vs data comparison."""
    PlotManager.compare_model_vs_data(*args, **kwargs)


def plot_loss_curve(*args: Any, **kwargs: Any) -> None:
    """Backward compatible wrapper for loss curve plotting."""
    PlotManager.plot_loss_curve(*args, **kwargs)


def plot_metric(*args: Any, **kwargs: Any) -> None:
    """Backward compatible wrapper for metric plotting."""
    PlotManager.plot_metric(*args, **kwargs)