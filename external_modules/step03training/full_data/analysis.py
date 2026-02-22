from typing import Any, Dict, List, Optional, Set, Tuple
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from shared.loaders.training_loader import training_loader


def plot_scatter_comparison(
    y_plot: np.ndarray, p_plot: np.ndarray, plot: bool = True
) -> None:
    _, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(
        y_plot, p_plot, alpha=0.7, s=30, edgecolors="k", linewidths=0.2, zorder=5
    )
    ymin = float(min(np.min(y_plot), np.min(p_plot)))
    ymax = float(max(np.max(y_plot), np.max(p_plot)))
    if not np.isfinite(ymin) or not np.isfinite(ymax):
        print("Could not determine plot range; skipping perfect-prediction line.")
    else:
        margin = max((ymax - ymin) * 0.02, 1e-06)
        ax.plot(
            [ymin - margin, ymax + margin],
            [ymin - margin, ymax + margin],
            "r--",
            label="perfect prediction",
            linewidth=2.0,
            zorder=10,
        )
        ax.set_xlim(ymin - margin, ymax + margin)
        ax.set_ylim(ymin - margin, ymax + margin)
        ax.set_aspect("equal", adjustable="box")
    ax.legend()
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted (sample)")
    ax.grid(True)
    if plot:
        plt.show()


def prepare_plot_data(y: Any, preds: Any) -> Tuple[np.ndarray, np.ndarray]:
    y_sample = np.asarray(y).ravel()
    preds = np.asarray(preds).ravel()
    mask = np.isfinite(y_sample) & np.isfinite(preds)
    if not mask.any():
        print("No finite prediction/data pairs to plot; skipping compare plot.")
        return (np.array([]), np.array([]))
    return (y_sample[mask], preds[mask])


def print_comparison_metrics(y: Any, preds: Any, metrics: Any) -> None:
    y_sample = np.asarray(y).ravel()
    preds = np.asarray(preds).ravel()
    mask_all = np.isfinite(y_sample) & np.isfinite(preds)
    if mask_all.any():
        y_eval = y_sample[mask_all]
        p_eval = preds[mask_all]
        sample_r2 = float(r2_score(y_eval, p_eval))
        print(f"Comparison metrics (sample): r2={sample_r2:.4f}, n={len(y_eval)}")
    if metrics is not None and "r2" in metrics:
        print(f"Stored metrics: r2={float(metrics['r2']):.4f}")


def compare_model_vs_data(
    x: np.ndarray,
    y: np.ndarray,
    plot: bool = True,
) -> None:
    """Compare a trained model vs data.

    This function requires a trained model to be supplied either as the `model`
    argument (in-memory estimator) or as a saved model file on disk. If no
    trained model is available, a RuntimeError is raised. This avoids silently
    re-training the model during comparison.
    """
    # Load data using the full pipeline (including filtering and interactions) to ensure consistency with trained model
    X = x
    x_sample = X[:500]
    # align y to the sample used for predictions
    y_sample = y[: len(x_sample)]
    model = training_loader.load_training_model()

    # Use the supplied/loaded model for predictions (do not retrain here).
    preds = model.predict(x_sample)
    # metrics may be available from saved diagnostics; attempt to load them
    metrics = None
    data = training_loader.load_training_model_diagnostics()
    if data is not None and "metrics" in data:
        metrics = data["metrics"]

    print_comparison_metrics(y_sample, preds, metrics)
    y_plot, p_plot = prepare_plot_data(y_sample, preds)
    if y_plot.size > 0 and p_plot.size > 0:
        plot_scatter_comparison(y_plot, p_plot, plot)


def plot_loss_curve(result_metrics: Dict[str, Any] | None = None) -> None:
    try:
        curves = None

        # Priority 1: passed result_metrics
        if (
            result_metrics is not None
            and "curves" in result_metrics
            and result_metrics["curves"] is not None
        ):
            curves = result_metrics["curves"]

        # Priority 2: load from disk
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
    """Callback to plot training progress in real-time inside Jupyter notebooks."""

    def __init__(
        self,
        update_interval: float = 5.0,
        min_time_before_first_plot: float = 10.0,
        save_path: Optional[str] = None,
    ):
        self.update_interval = update_interval
        self.min_time = min_time_before_first_plot
        self.save_path = save_path
        if self.save_path:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        self.start_time = time.time()
        self.last_plot_time = 0.0
        self.history: Dict[str, List[float]] = {}  # Generalized history storage

    def __call__(self, env: Any) -> None:
        now = time.time()
        elapsed = now - self.start_time

        # Collect metrics
        # env.evaluation_result_list looks like [('train', 'l2', 0.5, False), ('valid', 'l2', 0.6, False), ...]
        for data_name, eval_name, result, _ in env.evaluation_result_list:
            key = f"{data_name}_{eval_name}"
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(result)

        if elapsed < self.min_time:
            return

        if now - self.last_plot_time < self.update_interval:
            return

        self.last_plot_time = now

        try:
            # Group metrics by type (l2, rmse, l1)
            metrics_found: Set[str] = set()
            for key in self.history.keys():
                parts = key.split("_")
                if len(parts) >= 2:
                    metrics_found.add(parts[1])

            num_metrics = len(metrics_found)
            if num_metrics == 0:
                return

            fig, axes = plt.subplots(1, num_metrics, figsize=(5 * num_metrics, 5))
            if num_metrics == 1:
                axes = [axes]

            print(f"metrics found: {metrics_found}")

            for i, metric in enumerate(sorted(list(metrics_found))):
                ax = axes[i]
                if f"train_{metric}" in self.history:
                    data = self.history[f"train_{metric}"]
                    ax.plot(data, label=f"Train {metric}")
                    if data:
                        ax.annotate(
                            f"{data[-1]:.4f}",
                            xy=(len(data) - 1, data[-1]),
                            xytext=(5, 0),
                            textcoords="offset points",
                        )
                if f"valid_{metric}" in self.history:
                    data = self.history[f"valid_{metric}"]
                    ax.plot(data, label=f"Valid {metric}")
                    if data:
                        ax.annotate(
                            f"{data[-1]:.4f}",
                            xy=(len(data) - 1, data[-1]),
                            xytext=(5, 0),
                            textcoords="offset points",
                        )
                ax.set_xlabel("Iteration")
                ax.set_title(f"{metric.upper()} Progress")
                ax.legend()
                ax.grid(True)

            plt.suptitle(f"Training Progress (Elapsed: {elapsed:.1f}s)")
            plt.tight_layout()

            if self.save_path:
                try:
                    plt.savefig(self.save_path)
                except Exception:
                    pass

            plt.show()
        except ImportError:
            pass
        except Exception:
            pass
