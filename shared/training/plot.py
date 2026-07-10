from collections.abc import Mapping, Sequence
from typing import Any
import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt: Any = plt
from sklearn.metrics import r2_score
from statistics import mean, stdev
from scipy.special import softmax  # type: ignore[attr-defined]

import math

from ..loaders.training_loader import training_loader
from ..config import config
from .calibration import apply_score_calibration, extract_score_calibration



class PlotManager:
    """Manages all plotting functionality for model training and analysis."""

    # Metric direction mapping (shared across methods)
    METRIC_DIRECTIONS: dict[str, dict[str, bool]] = {
        "lambdarank": {"ndcg": True, "pairwise_accuracy": True, "score": True},
        "multiclass": {"multi_logloss": False, "multi_error": False},
        "binary": {"binary_logloss": False, "auc": True},
        "regression": {"l2": False, "rmse": False, "l1": False, "r2": True},
    }

    @staticmethod
    def _get_metric_direction(objective: str, metric: str) -> bool:
        """Get whether higher is better for a given metric."""
        return PlotManager.METRIC_DIRECTIONS[objective].get(metric, True)

    @staticmethod
    def _prepare_finite_data(y: Any, preds: Any) -> tuple[np.ndarray, np.ndarray]:
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
        min_size_px: float = 10.0,
        max_size_px: float = 100.0,
        power: float = 0.5,
        label_threshold: int = 10,
        title: str = "Actual vs Predicted (sample)",
        x_label: str = "Actual",
        y_label: str = "Predicted",
    ) -> None:
        """Plot actual vs predicted values with cluster sizes."""
        if not plot:
            return

        points = np.column_stack((y_plot, p_plot))
        unique_points, counts = np.unique(points, axis=0, return_counts=True)

        sizes = PlotManager._calculate_scatter_sizes(
            counts, min_size_px, max_size_px, power
        )

        _, ax = plt.subplots(figsize=(6, 4))
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
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.grid(True)

        if plot:
            plt.show()

    @staticmethod
    def plot_scatter_comparison_continuous(
        y_plot: np.ndarray,
        p_plot: np.ndarray,
        plot: bool = True,
        min_size_px: float = 10.0,
        max_size_px: float = 100.0,
        power: float = 0.5,
        label_threshold: int = 10,
        title: str = "Actual vs Predicted (continuous)",
        x_label: str = "Actual",
        y_label: str = "Predicted",
    ) -> None:
        """Plot actual vs predicted values with continuous data."""
        if not plot:
            return

        sizes = PlotManager._calculate_scatter_sizes(
            np.ones(len(y_plot)), min_size_px, max_size_px, power
        )

        _, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(
            y_plot,
            p_plot,
            s=sizes,
            alpha=0.85,
            edgecolors="none",
            zorder=5,
        )

        # Add labels for high-count points
        for x, y in zip(y_plot, p_plot):
            if round(x, 2) >= label_threshold:
                ax.annotate(
                    str(round(x, 2)),
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
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.grid(True)

        if plot:
            plt.show()

    @staticmethod
    def prepare_plot_data(y: Any, preds: Any) -> tuple[np.ndarray, np.ndarray]:
        """Prepare data for plotting by filtering out non-finite values."""
        y_sample, preds_sample = PlotManager._prepare_finite_data(y, preds)
        if y_sample.size == 0:
            print("No finite prediction/data pairs to plot; skipping compare plot.")
        return y_sample, preds_sample

    @staticmethod
    def print_comparison_metrics(
        y: Any,
        preds: Any,
        metrics: Any,
        objective: str | None = None,
        calibrated: bool = False,
    ) -> None:
        """Print comparison metrics between predictions and true values."""
        objective = objective or config["training"]["objective"]
        y_sample, p_sample = PlotManager._prepare_finite_data(y, preds)
        if y_sample.size > 0:
            sample_r2 = float(r2_score(y_sample, p_sample))
            label = "calibrated sample" if calibrated else "sample"
            print(
                f"Comparison metrics ({label}): r2={sample_r2:.4f}, n={len(y_sample)}"
            )
        if metrics is not None:
            if objective == "lambdarank":
                if "pairwise_accuracy" in metrics:
                    print(
                        f"Stored metrics: pairwise_accuracy={float(metrics['pairwise_accuracy']):.4f}"
                    )
                if (
                    "score" in metrics
                    and "primary_metric" in metrics
                    and metrics["primary_metric"] != "pairwise_accuracy"
                ):
                    try:
                        print(
                            f"Stored metrics: {metrics['primary_metric']}={float(metrics['score']):.4f}"
                        )
                    except Exception:
                        pass
            elif "r2" in metrics:
                print(f"Stored metrics: r2={float(metrics['r2']):.4f}")

    @staticmethod
    def compare_model_vs_data(
        x: np.ndarray, y: np.ndarray, plot: bool = True, limit: int = 100
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
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=UserWarning,
                    message=".*X does not have valid feature names.*",
                )
                preds = model.predict(x_sample)
        except Exception as e:
            print(f"Error during prediction: {e}")
            return

        diagnostics = training_loader.load_training_model_diagnostics()
        metrics = diagnostics if diagnostics is not None else None
        objective = config["training"]["objective"]

        calibration = extract_score_calibration(diagnostics)
        calibrated = False
        if objective == "lambdarank" and calibration is not None:
            preds = apply_score_calibration(preds, calibration)
            calibrated = True
        elif objective == "lambdarank":
            print(
                "Warning: no saved score calibration found; plotting raw ranker scores."
            )

        PlotManager.print_comparison_metrics(
            y_sample, preds, metrics, objective=objective, calibrated=calibrated
        )

        y_plot, p_plot = PlotManager.prepare_plot_data(y_sample, preds)

        if y_plot.size > 0 and p_plot.size > 0:
            if objective == "lambdarank":
                PlotManager.plot_scatter_comparison_continuous(
                    y_plot,
                    p_plot,
                    plot,
                    title="Actual vs Calibrated Predicted (ranking)",
                    x_label="Actual score",
                    y_label="Calibrated predicted score",
                )
            else:
                PlotManager.plot_scatter_comparison_continuous(y_plot, p_plot, plot)

    @staticmethod
    def _plot_metric_on_axes(
        ax: Any,
        metric_name: str,
        values: list[float],
        label: str,
        direction_higher: bool,
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
        axes: list[Any],
        current_metric: dict[str, list[float]],
        label: str = "Valid",
    ) -> None:
        """Plot individual metrics on subplots."""
        objective = config["training"]["objective"]
        for i, metric in enumerate(list(current_metric)):
            ax = axes[i]
            values = current_metric[metric]
            direction_higher = PlotManager._get_metric_direction(objective, metric)
            PlotManager._plot_metric_on_axes(
                ax, metric, values, label, direction_higher
            )

    @staticmethod
    def plot_loss_curve(result_metrics: dict[str, Any] | None = None) -> None:
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

    @staticmethod
    def plot_continuous_analysis(
        data_dict: Mapping[str, Sequence[tuple[float, float] | tuple[float, float, int]]],
        group_name: str,
        x_label: str,
        y_label: str,
        cols: int = 4,
        share_axes: bool = True,
    ) -> None:
        """
        Scatter plot grid for continuous metrics. Each subplot shows (value, score).

        Args:
            data_dict: {name: [(value, score), ...]}.
            group_name: Title for the whole figure.
            x_label: X-axis label.
            y_label: Y-axis label.
            cols: Number of columns in the subplot grid.
            share_axes: When True all subplots share the same axis limits.
        """
        titles = [t for t, pts in data_dict.items() if pts]
        if not titles:
            return
        n = len(titles)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5),
                                 sharex=share_axes, sharey=share_axes,
                                 squeeze=False)
        for i, title in enumerate(titles):
            ax = axes.flat[i]
            points = data_dict[title]
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            ax.scatter(x_coords, y_coords, color="blue", alpha=0.3, s=3, edgecolors="none")
            ax.set_title(title, fontsize=9)
            ax.grid(True, linestyle="--", alpha=0.3)
            if not share_axes:
                margin_x = (max(x_coords) - min(x_coords)) * 0.05 or 0.1
                margin_y = (max(y_coords) - min(y_coords)) * 0.05 or 0.1
                ax.set_xlim(min(x_coords) - margin_x, max(x_coords) + margin_x)
                ax.set_ylim(min(y_coords) - margin_y, max(y_coords) + margin_y)
        for i in range(n, rows * cols):
            axes.flat[i].axis("off")
        fig.supxlabel(x_label)
        fig.supylabel(y_label)
        fig.suptitle(group_name, fontsize=14)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_discrete_analysis(
        data_dict: dict[str, dict[str | int, list[float]]],
        group_name: str,
        x_label: str,
        y_label: str,
        cols: int = 4,
    ) -> None:
        """
        Plots discrete data in a grid of subplots with shared y-axis.

        Args:
            data_dict: dict[title, dict[x_value, list[y_values]]]
            group_name: A string representing the source dictionary.
            x_label: Label for the x-axis.
            y_label: Label for the y-axis.
            cols: Number of columns in the subplot grid.
        """
        titles = [t for t, d in data_dict.items() if d]
        if not titles:
            return
        n = len(titles)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4),
                                 sharey=True, squeeze=False)
        for i, title in enumerate(titles):
            ax = axes.flat[i]
            inner_dict = data_dict[title]
            x_coords, y_coords = [], []
            for x_val, y_list in inner_dict.items():
                for y_val in y_list:
                    x_coords.append(x_val)
                    y_coords.append(y_val)
            ax.scatter(x_coords, y_coords, color="orange", alpha=0.3, s=3, edgecolors="none")
            ax.set_title(title, fontsize=9)
            ax.grid(True, linestyle="--", alpha=0.3)
        for j in range(n, rows * cols):
            axes.flat[j].axis("off")
        fig.supxlabel(x_label)
        fig.supylabel(y_label)
        fig.suptitle(group_name, fontsize=14)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_aggregate_summary(
        data_dict: Mapping[str, Sequence[tuple[float, float] | tuple[float, float, int]]],
        group_name: str,
        value_label: str,
        top_percent: float = 0.10,
        limit: int = 0,
        ascending: bool = False,
    ) -> None:
        """
        Plots a sorted bar chart with usage thresholds and display limits.

        Args:
            data_dict: Dict of {name: [(weight, score), ...]}.
            group_name: Title of the data (e.g., 'Prompts').
            value_label: Metric name for Y-axis.
            top_percent: Initial filter: keep only the most-used X% of categories.
            limit: Final display filter: show only N elements. 0 shows all.
            ascending: If True, sorts worst-to-best. If False, best-to-worst.
        """
        if not data_dict:
            return

        # 1. Significance Filter: Keep only the top X% by usage count
        sorted_by_usage = sorted(
            data_dict.items(), key=lambda x: len(x[1]), reverse=True
        )
        usage_threshold = max(1, int(len(sorted_by_usage) * top_percent))
        frequent_data = sorted_by_usage[:usage_threshold]

        # 2. Calculate Statistics
        stats_list: list[dict[str, Any]] = []
        for name, points in frequent_data:
            scores = [p[1] for p in points]
            stats_list.append(
                {
                    "name": name,
                    "mean": mean(scores),
                    "std": stdev(scores) if len(scores) > 1 else 0.0,
                    "count": len(scores),
                }
            )

        # 3. Performance Sort (Best vs Worst)
        stats_list.sort(key=lambda x: x["mean"], reverse=not ascending)

        # 4. Display Limit (Slice the results)
        if limit > 0:
            stats_list = stats_list[:limit]

        if not stats_list:
            print("No data meets the current filters.")
            return

        # 5. Visualization
        labels = [
            (
                f"{s['name'][:30]}...\n(n={s['count']})"
                if len(s["name"]) > 35
                else f"{s['name']}\n(n={s['count']})"
            )
            for s in stats_list
        ]
        means = [s["mean"] for s in stats_list]
        errors = [s["std"] for s in stats_list]

        plt.figure(figsize=(14, 8))

        # Color logic: Green for good, Red for bad based on sorting intent
        cmap = (
            plt.cm.get_cmap("RdYlGn") if not ascending else plt.cm.get_cmap("RdYlGn_r")
        )
        colors = cmap([0.1 + (0.8 * (i / len(means))) for i in range(len(means))])

        bars = plt.bar(
            labels,
            means,
            yerr=errors,
            capsize=6,
            color=colors,
            edgecolor="black",
            alpha=0.8,
        )

        # Title adjustments based on filters
        sort_type = "Worst" if ascending else "Best"
        limit_text = f"Top {limit} " if limit > 0 else "All Significant "
        plt.title(
            f"{limit_text}{sort_type} {group_name}\n(Filter: Top {int(top_percent*100)}% by Usage)",
            fontsize=14,
            fontweight="bold",
        )

        plt.ylabel(f"Avg {value_label}", fontsize=12)
        plt.xticks(rotation=45, ha="right", fontsize=9)
        plt.grid(axis="y", linestyle=":", alpha=0.5)

        # Data Labels
        for bar in bars:
            h = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                h,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_individual_metrics(
        data_dict: dict[str, list[tuple[float, float]]], cols: int = 4, bins: int = 10
    ) -> None:
        """
        Plots Average Score (Y) vs Setting Value (X) for every metric.
        Bar width represents the sample size (count) for that bucket.
        """
        active_metrics = {k: v for k, v in data_dict.items() if v}
        keys = list(active_metrics.keys())
        num_plots = len(keys)

        if num_plots == 0:
            return

        rows = math.ceil(num_plots / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4),
                                 sharey=True)
        axes = np.array(axes).flatten()

        i = 0
        for i, key in enumerate(keys):
            ax = axes[i]
            x_raw = np.array([p[0] for p in active_metrics[key]])
            y_raw = np.array([p[1] for p in active_metrics[key]])

            # --- BINNING LOGIC ---
            # If discrete (few unique X), use exact values. If continuous, create 'bins' ranges.
            unique_x = np.unique(x_raw)
            if len(unique_x) > bins:
                # Create ranges for continuous data
                min_x, max_x = x_raw.min(), x_raw.max()
                bin_edges = np.linspace(min_x, max_x, bins + 1)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                bin_indices = np.digitize(x_raw, bin_edges[:-1])
            else:
                bin_centers = unique_x
                bin_indices = np.searchsorted(unique_x, x_raw) + 1

            # --- AGGREGATE STATS ---
            means, stds, counts = [], [], []
            actual_centers = []

            for b_idx in range(1, len(bin_centers) + 1):
                mask = bin_indices == b_idx
                if not np.any(mask):
                    continue

                group_y = y_raw[mask]
                means.append(mean(group_y))
                stds.append(stdev(group_y) if len(group_y) > 1 else 0.0)
                counts.append(len(group_y))
                actual_centers.append(bin_centers[b_idx - 1])

            # --- VARIABLE WIDTH CALCULATION ---
            # Normalize counts to a reasonable width so they don't overlap
            counts = np.array(counts)
            if len(actual_centers) > 1:
                # Base width is 80% of the distance between centers
                max_width = (actual_centers[1] - actual_centers[0]) * 0.8
                widths = (counts / counts.max()) * max_width
            else:
                widths = [0.5]  # Default for single-column data

            # --- PLOTTING ---
            _ = ax.bar(
                actual_centers,
                means,
                yerr=stds,
                width=widths,
                capsize=5,
                color="skyblue",
                edgecolor="navy",
                alpha=0.7,
            )

            ax.set_title(
                f"{key}\n(Width = Sample Size)", fontsize=11, fontweight="bold"
            )
            ax.set_xlabel("Setting Value")
            ax.set_ylabel("Avg Score (± Std Dev)")
            ax.grid(axis="y", linestyle=":", alpha=0.6)

        # Cleanup
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_discrete_object_analysis(
        discrete_data: dict[str, dict[str | int, list[float]]],
        title_prefix: str = "Discrete Analysis",
        cols: int = 4,
    ) -> None:
        """
        Analyzes a nested discrete data structure in a grid of bar charts with shared y-axis.
        Structure: { "Metric": { "Category": [score1, score2...] } }
        """
        metrics = [(k, v) for k, v in discrete_data.items() if v]
        if not metrics:
            return
        n = len(metrics)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4),
                                 sharey=True, squeeze=False)
        for i, (metric_name, categories) in enumerate(metrics):
            ax = axes.flat[i]
            labels: list[str] = []
            means: list[float] = []
            errors: list[float] = []
            counts: list[int] = []
            sorted_keys = sorted(
                categories.keys(), key=lambda x: (isinstance(x, str), x)
            )
            for cat_key in sorted_keys:
                scores = categories[cat_key]
                if not scores:
                    continue
                labels.append(str(cat_key))
                means.append(mean(scores))
                errors.append(stdev(scores) if len(scores) > 1 else 0.0)
                counts.append(len(scores))
            if not labels:
                ax.axis("off")
                continue
            counts_array = np.array(counts)
            widths = (counts_array / counts_array.max()) * 0.8
            colors = plt.cm.get_cmap("plasma")(np.linspace(0.2, 0.6, len(labels)))
            bars = ax.bar(labels, means, yerr=errors, width=widths,
                          capsize=4, color=colors, edgecolor="black", alpha=0.8)
            ax.set_title(metric_name, fontsize=9)
            ax.grid(axis="y", linestyle=":", alpha=0.3)
            for bar, count in zip(bars, counts):
                ax.text(bar.get_x() + bar.get_width() / 2, 0.02,
                        f"n={count}", ha="center", va="bottom",
                        fontsize=7, color="white", fontweight="bold")
        for j in range(n, rows * cols):
            axes.flat[j].axis("off")
        fig.supylabel("Average Score (± Std Dev)")
        fig.suptitle(title_prefix, fontsize=14)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def prepare_face_data(
        text_data: list[dict[str, Any]],
        scores: Sequence[float],
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[float], list[float], list[float], list[float], list[float], list[float], int]:
        AGE_LABELS = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]
        GENDER_LABELS = ["Female", "Male"]
        RACE_LABELS = ["Black", "East Asian", "Indian", "Latino_Hispanic", "Middle Eastern", "Southeast Asian", "White"]

        face_logit_rows: list[dict[str, Any]] = []
        bbox_rows: list[dict[str, Any]] = []
        pose_score: list[float] = []
        no_pose_score: list[float] = []
        lh_score: list[float] = []
        no_lh_score: list[float] = []
        rh_score: list[float] = []
        no_rh_score: list[float] = []

        for i in range(len(text_data)):
            outer = text_data[i]
            d = outer[next(iter(outer))]
            s = float(scores[i])

            fl = d.get("face_logits")
            if fl and len(fl) >= 18:
                row = {}
                for j, lbl in enumerate(AGE_LABELS):
                    row[f"age_{j}_{lbl}"] = fl[j]
                for j, lbl in enumerate(GENDER_LABELS):
                    row[f"gender_{j}_{lbl}"] = fl[9 + j]
                for j, lbl in enumerate(RACE_LABELS):
                    row[f"race_{j}_{lbl}"] = fl[11 + j]
                age_sm = softmax(fl[:9])
                gender_sm = softmax(fl[9:11])
                race_sm = softmax(fl[11:18])
                for j, lbl in enumerate(AGE_LABELS):
                    row[f"age_sm_{j}_{lbl}"] = age_sm[j]
                for j, lbl in enumerate(GENDER_LABELS):
                    row[f"gender_sm_{j}_{lbl}"] = gender_sm[j]
                for j, lbl in enumerate(RACE_LABELS):
                    row[f"race_sm_{j}_{lbl}"] = race_sm[j]
                row["score"] = s
                face_logit_rows.append(row)

            bbox = d.get("face_bbox")
            if bbox and len(bbox) > 0:
                b = bbox[0]
                bbox_rows.append({
                    "x": b[0], "y": b[1], "w": b[2], "h": b[3],
                    "conf": b[4] if len(b) > 4 else 1.0,
                    "score": s,
                })

            if d.get("body_pose"):
                pose_score.append(s)
            else:
                no_pose_score.append(s)
            if d.get("left_hand"):
                lh_score.append(s)
            else:
                no_lh_score.append(s)
            if d.get("right_hand"):
                rh_score.append(s)
            else:
                no_rh_score.append(s)

        df_face = pd.DataFrame(face_logit_rows)
        df_bbox = pd.DataFrame(bbox_rows)
        n = len(text_data)
        return df_face, df_bbox, pose_score, no_pose_score, lh_score, no_lh_score, rh_score, no_rh_score, n

    @staticmethod
    def plot_face_bbox(df_bbox: pd.DataFrame) -> None:
        if len(df_bbox) == 0:
            print("No face bbox data")
            return
        # Position scatter
        _, ax = plt.subplots(figsize=(6, 6))
        sc = ax.scatter(df_bbox["x"], df_bbox["y"], c=df_bbox["score"],
                        s=5, alpha=0.5, cmap="RdYlGn")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.invert_yaxis()
        ax.set_xlabel("Face x (fraction of image width, 0=left, 1=right)")
        ax.set_ylabel("Face y (fraction of image height, 0=top, 1=bottom)")
        ax.set_title("Face Position in Image (color = score)")
        plt.colorbar(sc, ax=ax, label="Score")
        plt.tight_layout()
        plt.show()
        # Size vs score
        _, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
        axes[0].scatter(df_bbox["w"], df_bbox["score"], s=3, alpha=0.3, c="steelblue")
        axes[0].set_xlabel("Face width (fraction of image width)")
        axes[0].set_title("Width vs Score")
        axes[1].scatter(df_bbox["h"], df_bbox["score"], s=3, alpha=0.3, c="orange")
        axes[1].set_xlabel("Face height (fraction of image height)")
        axes[1].set_title("Height vs Score")
        axes[0].set_ylabel("Score")
        for ax in axes:
            ax.axhline(0, color="gray", ls=":", alpha=0.3)
            ax.axvline(0, color="gray", ls=":", alpha=0.3)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_positional_data(
        pos_data: dict[str, dict[str, list[float]]],
        group_name: str = "Positional Data",
        cols: int = 4,
        invert_y: bool = True,
    ) -> None:
        """
        Plots positional data in a grid.
        Each entry: {name -> {"x": [...], "y": [...], "score": [...]}}
        Shows x vs y colored by score (inverted y for image coords).
        If "w" and "h" are present, also adds size vs score subplots.
        """
        names = [k for k, v in pos_data.items() if len(v.get("x", [])) > 0]
        if not names:
            return
        n_pos = len(names)
        rows = (n_pos + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4.5),
                                 squeeze=False)
        for i, name in enumerate(names):
            ax = axes.flat[i]
            d = pos_data[name]
            sc = ax.scatter(d["x"], d["y"], c=d["score"], s=5, alpha=0.5,
                            cmap="RdYlGn")
            if invert_y:
                ax.invert_yaxis()
            ax.set_title(name, fontsize=9)
            ax.set_xlabel("x"); ax.set_ylabel("y")
            plt.colorbar(sc, ax=ax, label="Score")
        for j in range(n_pos, rows * cols):
            axes.flat[j].axis("off")
        fig.suptitle(group_name, fontsize=14)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_positional_bbox(
        pos_data: dict[str, dict[str, list[float]]],
        group_name: str = "Bounding Boxes",
        cols: int = 4,
        invert_y: bool = True,
        alpha: float = 0.3,
    ) -> None:
        """
        Draws colored bounding boxes for positional entries that have x, y, w, h.
        Each box is colored by its score (RdYlGn). One subplot per entry.
        """
        from matplotlib.patches import Rectangle
        from matplotlib.colors import Normalize

        names = [k for k, v in pos_data.items()
                 if len(v.get("x", [])) > 0 and "w" in v and "h" in v]
        if not names:
            return
        n_pos = len(names)
        rows = (n_pos + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4.5),
                                 squeeze=False)
        for i, name in enumerate(names):
            ax = axes.flat[i]
            d = pos_data[name]
            scores = d["score"]
            vmin, vmax = min(scores), max(scores)
            norm = Normalize(vmin=vmin, vmax=vmax) if vmax > vmin else Normalize(vmin=0, vmax=1)
            color_cmap = plt.get_cmap("RdYlGn")
            for j in range(len(d["x"])):
                color = color_cmap(norm(scores[j]))
                rect = Rectangle(
                    (d["x"][j], d["y"][j]), d["w"][j], d["h"][j],
                    linewidth=1.0, edgecolor=color, facecolor="none",
                )
                ax.add_patch(rect)
            ax.set_xlim(0, 1)
            if invert_y:
                ax.set_ylim(1, 0)
            else:
                ax.set_ylim(0, 1)
            ax.set_aspect('equal')
            ax.set_title(name, fontsize=9)
            ax.set_xlabel("x"); ax.set_ylabel("y")
            sm = plt.cm.ScalarMappable(norm=norm, cmap="RdYlGn")
            sm.set_array([])
            plt.colorbar(sm, ax=ax, label="Score")
        for j in range(n_pos, rows * cols):
            axes.flat[j].axis("off")
        fig.suptitle(group_name, fontsize=14)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_detection_presence(
        pose_score: list[float], no_pose_score: list[float],
        lh_score: list[float], no_lh_score: list[float],
        rh_score: list[float], no_rh_score: list[float],
        n: int,
    ) -> None:
        from scipy.stats import mannwhitneyu  # type: ignore[attr-defined]
        fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
        detect_pairs = [
            ("Body Pose", pose_score, no_pose_score),
            ("Left Hand", lh_score, no_lh_score),
            ("Right Hand", rh_score, no_rh_score),
        ]
        for ax, (name, yes, no) in zip(axes, detect_pairs):
            bp = ax.boxplot([yes, no], labels=[f"Detected\n(n={len(yes)})", f"Not\n(n={len(no)})"],
                            patch_artist=True)
            bp["boxes"][0].set_facecolor("steelblue")
            bp["boxes"][1].set_facecolor("lightcoral")
            ax.set_title(f"{name}")
            ax.axhline(0, color="gray", ls=":", alpha=0.3)
            if len(yes) > 0 and len(no) > 0:
                _, p = mannwhitneyu(yes, no, alternative="two-sided")
                ax.text(0.5, 0.95, f"MW p={p:.4f}", transform=ax.transAxes,
                        ha="center", fontsize=9, style="italic")
        axes[0].set_ylabel("Score")
        fig.suptitle("Detection Presence vs Score")
        plt.tight_layout()
        plt.show()
    """Callback to plot training progress only at the end of training."""

    def __init__(
        self, save_path: str | None = None, frequency: int = 30, status_bar: Any = None
    ) -> None:
        """Initialize callback to plot only final results."""
        self.save_path = save_path
        if self.save_path:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        self.last_plot_time = time.time()
        self.history: dict[str, dict[str, list[float]]] = {
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
            if self.status_bar is not None:
                with self.status_bar:
                    self.status_bar.set_description("Plotting final results...")
                    self.plot_final_results()
            else:
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
