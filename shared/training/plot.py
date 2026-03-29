from typing import Any, Dict, List, Tuple, Optional, Union
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from statistics import mean, stdev

import seaborn as sns
import pandas as pd
import math

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
        min_size_px: float = 10.0,
        max_size_px: float = 100.0,
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
    def plot_scatter_comparison_continuous(
        y_plot: np.ndarray,
        p_plot: np.ndarray,
        plot: bool = True,
        min_size_px: float = 10.0,
        max_size_px: float = 100.0,
        power: float = 0.5,
        label_threshold: int = 10,
    ) -> None:
        """Plot actual vs predicted values with continuous data."""
        

        sizes = PlotManager._calculate_scatter_sizes(
            np.ones(len(y_plot)), min_size_px, max_size_px, power
        )

        fig, ax = plt.subplots(figsize=(6, 4))
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
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Actual vs Predicted (continuous)")
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
            PlotManager.plot_scatter_comparison_continuous(y_plot, p_plot, plot)

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
    
    
    @staticmethod
    def plot_continuous_analysis(data_dict: Dict[str, List[Tuple[float, float]]], group_name: str, x_label:str,y_label:str):
        """
        Iterates through a dictionary and creates individual scatter plots.
        
        Args:
            data_dict: The dictionary containing category names (str) and data points (List[Tuple]).
            group_name: A string representing the source dictionary (e.g., 'lora_data').
        """
        for title, points in data_dict.items():
            if not points:
                print(f"No data for {title}")
                continue
                
            # Unpack the list of tuples [(x1, y1), (x2, y2), ...] into two lists
            # x_coords = [x1, x2, ...], y_coords = [y1, y2, ...]
            x_coords, y_coords = zip(*points)
            
            plt.figure(figsize=(10, 6))
            plt.scatter(x_coords, y_coords, color='blue', alpha=0.7, edgecolors='black')
            
            # Setting the title and labels
            plt.title(f"{group_name}: {title}", fontsize=14)
            plt.xlabel(x_label, fontsize=12)
            plt.ylabel(y_label, fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.6)
            
            # Display or save the plot
            plt.tight_layout()
            plt.show()
            
    @staticmethod
    def plot_discrete_analysis(data_dict: Dict[str, Dict[str | int, List[float]]], group_name: str, x_label: str, y_label: str):
        """
        Iterates through a nested dictionary and creates individual scatter plots for discrete data.
        
        Args:
            data_dict: Dict[title, Dict[x_value, List[y_values]]]
            group_name: A string representing the source dictionary (e.g., 'discrete_data').
            x_label: Label for the x-axis.
            y_label: Label for the y-axis.
        """
        for title, inner_dict in data_dict.items():
            if not inner_dict:
                print(f"No data for {title}")
                continue
                
            x_coords = []
            y_coords = []
            
            # Flatten the nested structure into x, y coordinates
            for x_val, y_list in inner_dict.items():
                for y_val in y_list:
                    x_coords.append(x_val)
                    y_coords.append(y_val)
            
            plt.figure(figsize=(10, 6))
            plt.scatter(x_coords, y_coords, color='orange', alpha=0.7, edgecolors='black')
            
            # Setting the title and labels using your new parameters
            plt.title(f"{group_name}: {title}", fontsize=14)
            plt.xlabel(x_label, fontsize=12)
            plt.ylabel(y_label, fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.6)
            
            # Display or save the plot
            plt.tight_layout()
            plt.show()


    @staticmethod
    def plot_aggregate_summary(
        data_dict: Dict[str, List[Tuple[float, float]]], 
        group_name: str, 
        value_label: str,
        top_percent: float = 0.10,
        limit: int = 0,
        ascending: bool = False
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
        sorted_by_usage = sorted(data_dict.items(), key=lambda x: len(x[1]), reverse=True)
        usage_threshold = max(1, int(len(sorted_by_usage) * top_percent))
        frequent_data = sorted_by_usage[:usage_threshold]

        # 2. Calculate Statistics
        stats_list: List[Dict[str, Any]] = []
        for name, points in frequent_data:
            scores = [p[1] for p in points]
            stats_list.append({
                "name": name,
                "mean": mean(scores),
                "std": stdev(scores) if len(scores) > 1 else 0.0,
                "count": len(scores)
            })

        # 3. Performance Sort (Best vs Worst)
        stats_list.sort(key=lambda x: x["mean"], reverse=not ascending)

        # 4. Display Limit (Slice the results)
        if limit > 0:
            stats_list = stats_list[:limit]

        if not stats_list:
            print("No data meets the current filters.")
            return

        # 5. Visualization
        labels = [f"{s['name'][:30]}...\n(n={s['count']})" if len(s['name']) > 35 
                  else f"{s['name']}\n(n={s['count']})" for s in stats_list]
        means = [s["mean"] for s in stats_list]
        errors = [s["std"] for s in stats_list]

        plt.figure(figsize=(14, 8))
        
        # Color logic: Green for good, Red for bad based on sorting intent
        cmap = plt.cm.RdYlGn if not ascending else plt.cm.RdYlGn_r
        colors = cmap([0.1 + (0.8 * (i / len(means))) for i in range(len(means))])

        bars = plt.bar(labels, means, yerr=errors, capsize=6, 
                       color=colors, edgecolor='black', alpha=0.8)

        # Title adjustments based on filters
        sort_type = "Worst" if ascending else "Best"
        limit_text = f"Top {limit} " if limit > 0 else "All Significant "
        plt.title(f"{limit_text}{sort_type} {group_name}\n(Filter: Top {int(top_percent*100)}% by Usage)", 
                  fontsize=14, fontweight='bold')
        
        plt.ylabel(f"Avg {value_label}", fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.grid(axis='y', linestyle=':', alpha=0.5)

        # Data Labels
        for bar in bars:
            h = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, h, f'{h:.3f}', 
                     ha='center', va='bottom', fontsize=8, fontweight='bold')

        plt.tight_layout()
        plt.show()
        
    @staticmethod
    def plot_individual_metrics(data_dict: Dict[str, List[Tuple[float, float]]], cols: int = 4, bins: int = 10):
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
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
        axes = np.array(axes).flatten()

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
                mask = (bin_indices == b_idx)
                if not np.any(mask): continue
                
                group_y = y_raw[mask]
                means.append(mean(group_y))
                stds.append(stdev(group_y) if len(group_y) > 1 else 0.0)
                counts.append(len(group_y))
                actual_centers.append(bin_centers[b_idx-1])

            # --- VARIABLE WIDTH CALCULATION ---
            # Normalize counts to a reasonable width so they don't overlap
            counts = np.array(counts)
            if len(actual_centers) > 1:
                # Base width is 80% of the distance between centers
                max_width = (actual_centers[1] - actual_centers[0]) * 0.8 
                widths = (counts / counts.max()) * max_width
            else:
                widths = [0.5] # Default for single-column data

            # --- PLOTTING ---
            bars = ax.bar(actual_centers, means, yerr=stds, width=widths, 
                          capsize=5, color='skyblue', edgecolor='navy', alpha=0.7)

            ax.set_title(f"{key}\n(Width = Sample Size)", fontsize=11, fontweight='bold')
            ax.set_xlabel("Setting Value")
            ax.set_ylabel("Avg Score (± Std Dev)")
            ax.grid(axis='y', linestyle=':', alpha=0.6)

        # Cleanup
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()
       
       
       
    @staticmethod
    def plot_discrete_object_analysis(
        discrete_data: Dict[str, Dict[Union[str, int], List[float]]], 
        title_prefix: str = "Discrete Analysis"
    ) -> None:
        """
        Analyzes a nested discrete data structure.
        Structure: { "Metric": { "Category": [score1, score2...] } }
        """
        for metric_name, categories in discrete_data.items():
            if not categories:
                continue

            labels: List[str] = []
            means: List[float] = []
            errors: List[float] = []
            counts: List[int] = []

            # Sort categories (ints numerically, strings alphabetically)
            sorted_keys = sorted(categories.keys(), key=lambda x: (isinstance(x, str), x))

            for cat_key in sorted_keys:
                scores = categories[cat_key]
                if not scores:
                    continue
                
                labels.append(str(cat_key))
                means.append(mean(scores))
                errors.append(stdev(scores) if len(scores) > 1 else 0.0)
                counts.append(len(scores))

            # --- WIDTH CALCULATION ---
            counts_array = np.array(counts)
            # Width is proportional to the number of samples in that specific category
            # Max width 0.8 to keep visual separation
            widths = (counts_array / counts_array.max()) * 0.8

            plt.figure(figsize=(12, 6))
            
            # Use a distinctive color for the bars
            colors = plt.cm.plasma(np.linspace(0.2, 0.6, len(labels)))

            bars = plt.bar(
                labels, 
                means, 
                yerr=errors, 
                width=widths, 
                capsize=8, 
                color=colors, 
                edgecolor='black', 
                alpha=0.8
            )

            plt.title(f"{title_prefix}: {metric_name}\n(Width = Sample Count)", fontsize=14, fontweight='bold')
            plt.ylabel("Average Score (± Std Dev)", fontsize=12)
            plt.xlabel("Category / Setting", fontsize=12)
            plt.grid(axis='y', linestyle=':', alpha=0.7)

            # Add count overlay for precision
            for bar, count in zip(bars, counts):
                plt.text(
                    bar.get_x() + bar.get_width()/2, 
                    0.02, 
                    f"n={count}", 
                    ha='center', 
                    va='bottom', 
                    fontsize=9, 
                    color='white', 
                    fontweight='bold'
                )

            plt.tight_layout()
            plt.show() 
        
        
        



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