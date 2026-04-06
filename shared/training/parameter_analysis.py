"""
Parameter Analysis Module
Analyzes relationships between parameters/terms and image scores.
Generates correlation statistics and visualizations.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any

try:
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use("Agg")  # Use non-interactive backend
    from matplotlib.colors import Normalize

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("Warning: matplotlib not available - analysis will be limited")
    MATPLOTLIB_AVAILABLE = False

try:
    from sklearn.preprocessing import MinMaxScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    print("Warning: scikit-learn not available - normalization will be basic")
    SKLEARN_AVAILABLE = False


class ParameterAnalyzer:
    """Analyzes vector data for parameter and term correlations with scores."""

    def __init__(
        self,
        vectors_data: list[dict[str, Any]],
        text_data: list[dict[str, Any]],
        output_dir: str = "output/analysis",
    ):
        """
        Initialize analyzer with vector and text data.

        Args:
            vectors_data: List of vector entries with parameters and scores
            text_data: List of text entries with extracted terms
            output_dir: Directory to save analysis output
        """
        self.vectors = vectors_data
        self.text_data = text_data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Extract scores for later use
        self.scores = np.array([v.get("score", 0) for v in vectors_data])

    def analyze_all(self) -> None:
        """Run all analyses and generate report."""
        print("Starting parameter analysis...")

        if MATPLOTLIB_AVAILABLE:
            print("Analyzing parameter relationships...")
            self.analyze_parameter_pairs()
            self.analyze_term_correlations()
            print("✓ Parameter analysis complete")
        else:
            print("⚠ matplotlib not available - skipping visualization")

        print("Generating analysis report...")
        self.generate_report()

        print(f"✓ Analysis complete! Output saved to {self.output_dir}")

    def analyze_parameter_pairs(self) -> None:
        """Analyze relationships between parameter pairs and scores."""
        if not MATPLOTLIB_AVAILABLE:
            return

        print("  - Extracting parameters...")

        # Extract common parameters from vectors
        steps_list = []
        cfg_list = []
        lora_weight_list = []
        sampler_list = []
        scheduler_list = []
        model_list = []

        for vector in self.vectors:
            # Extract parameters (these field names depend on your actual data structure)
            # Adjust these keys based on your actual vector format
            steps = vector.get("generation_params", {}).get("steps", 0)
            cfg = vector.get("generation_params", {}).get("cfg_scale", 0)
            lora_w = vector.get("generation_params", {}).get("lora_weight", 0)
            sampler = vector.get("generation_params", {}).get("sampler_name", "unknown")
            scheduler = vector.get("generation_params", {}).get("scheduler", "unknown")
            model = vector.get("generation_params", {}).get("model", "unknown")

            if steps > 0:
                steps_list.append(steps)
            if cfg > 0:
                cfg_list.append(cfg)
            if lora_w > 0:
                lora_weight_list.append(lora_w)
            if sampler != "unknown":
                sampler_list.append(sampler)
            if scheduler != "unknown":
                scheduler_list.append(scheduler)
            if model != "unknown":
                model_list.append(model)

        print(f"    Found {len(steps_list)} entries with steps parameter")
        print(f"    Found {len(cfg_list)} entries with CFG parameter")
        print(f"    Found {len(lora_weight_list)} entries with LORA weight")

        # Create scatter plots for numeric parameters
        if len(steps_list) > 1:
            print("  - Creating Steps vs Score plot...")
            self._create_scatter(
                np.array(steps_list),
                self.scores[: len(steps_list)],
                self.scores[: len(steps_list)],
                "steps_vs_score",
                "Sampling Steps",
                "Score",
                normalize=True,
            )

        if len(cfg_list) > 1:
            print("  - Creating CFG vs Score plot...")
            self._create_scatter(
                np.array(cfg_list),
                self.scores[: len(cfg_list)],
                self.scores[: len(cfg_list)],
                "cfg_vs_score",
                "CFG Scale",
                "Score",
                normalize=True,
            )

        if len(steps_list) > 1 and len(cfg_list) > 1:
            min_len = min(len(steps_list), len(cfg_list))
            print("  - Creating Steps vs CFG plot...")
            self._create_2d_scatter(
                np.array(steps_list[:min_len]),
                np.array(cfg_list[:min_len]),
                self.scores[:min_len],
                "steps_vs_cfg",
                "Sampling Steps",
                "CFG Scale",
                "Score",
            )

        # Analyze categorical parameters
        if sampler_list:
            print("  - Analyzing sampler correlations...")
            sampler_scores = self._get_category_scores(sampler_list)
            self._save_category_stats("sampler_stats.json", sampler_scores)

        if scheduler_list:
            print("  - Analyzing scheduler correlations...")
            scheduler_scores = self._get_category_scores(scheduler_list)
            self._save_category_stats("scheduler_stats.json", scheduler_scores)

    def analyze_term_correlations(self) -> None:
        """Analyze relationship between terms and scores."""
        if not MATPLOTLIB_AVAILABLE:
            return

        print("  - Extracting term correlations...")
        term_scores: dict[str, list[float]] = {}

        for idx, text_entry in enumerate(self.text_data):
            if idx >= len(self.scores):
                continue

            score = self.scores[idx]

            # Extract positive and negative terms
            pos_terms = text_entry.get("positive_terms", [])
            neg_terms = text_entry.get("negative_terms", [])

            for term_data in pos_terms:
                if isinstance(term_data, (list, tuple)):
                    term, weight = term_data[0], (
                        term_data[1] if len(term_data) > 1 else 1.0
                    )
                else:
                    term, weight = str(term_data), 1.0

                if term not in term_scores:
                    term_scores[term] = []
                term_scores[term].append(score * weight)

            for term_data in neg_terms:
                if isinstance(term_data, (list, tuple)):
                    term, weight = term_data[0], (
                        term_data[1] if len(term_data) > 1 else 1.0
                    )
                else:
                    term, weight = str(term_data), 1.0

                if term not in term_scores:
                    term_scores[term] = []
                term_scores[term].append(score * (1 - weight))

        # Calculate statistics per term
        term_stats: dict[str, dict[str, float]] = {}
        for term, scores in term_scores.items():
            if len(scores) > 0:
                term_stats[term] = {
                    "avg_score": float(np.mean(scores)),
                    "std_dev": float(np.std(scores)),
                    "count": len(scores),
                    "max_score": float(np.max(scores)),
                    "min_score": float(np.min(scores)),
                }

        # Sort by average score
        sorted_terms = sorted(
            term_stats.items(), key=lambda x: x[1]["avg_score"], reverse=True
        )

        print(f"  - Found {len(term_stats)} unique terms")
        print(f"  - Top positive terms: {[t[0] for t in sorted_terms[:5]]}")

        # Save top correlations
        with open(self.output_dir / "term_correlations.json", "w") as f:
            json.dump(
                {
                    "top_positive_terms": sorted_terms[:50],
                    "bottom_terms": sorted_terms[-50:],
                    "total_unique_terms": len(term_stats),
                    "summary": {
                        "terms_analyzed": len(term_stats),
                        "avg_score_across_all": float(np.mean(self.scores)),
                    },
                },
                f,
                indent=2,
            )

    def _create_scatter(
        self,
        x: np.ndarray,
        y: np.ndarray,
        colors: np.ndarray,
        name: str,
        xlabel: str,
        ylabel: str,
        normalize: bool = False,
    ) -> None:
        """Create 1D scatter plot with color mapping."""
        if not MATPLOTLIB_AVAILABLE:
            return

        try:
            fig, ax = plt.subplots(figsize=(10, 8))

            # Normalize x if requested
            if normalize and SKLEARN_AVAILABLE:
                scaler = MinMaxScaler()
                x_plot = scaler.fit_transform(x.reshape(-1, 1)).flatten()
            else:
                x_plot = x

            scatter = ax.scatter(
                x_plot, y, c=colors, cmap="viridis", s=50, alpha=0.6, edgecolors="k"
            )
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_title(f"{xlabel} vs {ylabel}", fontsize=14, fontweight="bold")

            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label("Score", fontsize=12)

            plt.tight_layout()
            output_file = self.output_dir / f"{name}.png"
            plt.savefig(output_file, dpi=150, bbox_inches="tight")
            plt.close()

            print(f"    ✓ Saved {output_file.name}")
        except Exception as e:
            print(f"    ✗ Error creating scatter plot: {e}")

    def _create_2d_scatter(
        self,
        x: np.ndarray,
        y: np.ndarray,
        colors: np.ndarray,
        name: str,
        xlabel: str,
        ylabel: str,
        zlabel: str,
    ) -> None:
        """Create 2D scatter plot with color coding for Z dimension."""
        if not MATPLOTLIB_AVAILABLE:
            return

        try:
            fig, ax = plt.subplots(figsize=(12, 9))

            scatter = ax.scatter(
                x,
                y,
                c=colors,
                cmap="coolwarm",
                s=100,
                alpha=0.7,
                edgecolors="k",
                linewidth=0.5,
            )
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_title(
                f"{xlabel} vs {ylabel} (colored by {zlabel})",
                fontsize=14,
                fontweight="bold",
            )

            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(zlabel, fontsize=12)

            # Add correlation coefficient
            if len(x) > 1:
                corr = np.corrcoef(x, y)[0, 1]
                ax.text(
                    0.05,
                    0.95,
                    f"Correlation: {corr:.3f}",
                    transform=ax.transAxes,
                    fontsize=11,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
                )

            plt.tight_layout()
            output_file = self.output_dir / f"{name}.png"
            plt.savefig(output_file, dpi=150, bbox_inches="tight")
            plt.close()

            print(f"    ✓ Saved {output_file.name}")
        except Exception as e:
            print(f"    ✗ Error creating 2D scatter plot: {e}")

    def _get_category_scores(self, categories: list[str]) -> dict[str, list[float]]:
        """Get scores by category."""
        category_scores: dict[str, list[float]] = {}
        for cat, score in zip(categories, self.scores[: len(categories)]):
            if cat not in category_scores:
                category_scores[cat] = []
            category_scores[cat].append(float(score))
        return category_scores

    def _save_category_stats(
        self, filename: str, category_scores: dict[str, list[float]]
    ) -> None:
        """Save statistics for categorical data."""
        stats = {}
        for category, scores in category_scores.items():
            if len(scores) > 0:
                stats[category] = {
                    "avg_score": float(np.mean(scores)),
                    "std_dev": float(np.std(scores)),
                    "count": len(scores),
                    "max": float(np.max(scores)),
                    "min": float(np.min(scores)),
                }

        with open(self.output_dir / filename, "w") as f:
            json.dump(stats, f, indent=2)

    def generate_report(self) -> None:
        """Generate summary analysis report."""
        report = f"""# Parameter Analysis Report

## Summary Statistics
- **Total images analyzed**: {len(self.vectors)}
- **Average score**: {np.mean(self.scores):.2f}
- **Std Dev**: {np.std(self.scores):.2f}
- **Min score**: {np.min(self.scores):.2f}
- **Max score**: {np.max(self.scores):.2f}

## Analysis Components

### 1. Parameter Relationships
See the generated PNG files for 2D scatter plots showing:
- Steps vs Score
- CFG Scale vs Score
- Steps vs CFG (colored by score)

### 2. Term Correlations
See `term_correlations.json` for:
- Top 50 positive terms (highest average scores)
- Bottom 50 terms (lowest average scores)
- Statistical metrics per term

### 3. Categorical Analysis
Review JSON files for:
- Sampler correlations: `sampler_stats.json`
- Scheduler correlations: `scheduler_stats.json`

## Interpretation Guide

### Reading Scatter Plots
- **X-axis**: Parameter value
- **Y-axis**: Image score
- **Color**: Score intensity (brighter = higher score)
- **Clustering**: Dense clusters indicate parameter-score relationships

### Optimal Parameters
Look for regions with:
1. High score density (many points at high Y values)
2. Consistent coloring (purple/bright colors dominating)
3. Positive correlation (upward trend)

## Recommendations

1. **Experiment Focus**: Test parameters in high-density, high-score regions
2. **Term Strategy**: Use positively correlated terms more frequently
3. **Avoid**: Parameters and terms in low-score regions
4. **Combinations**: Test combinations that appear in brighter clusters

## Generated Files

- `steps_vs_score.png` - Sampling steps analysis
- `cfg_vs_score.png` - CFG scale analysis
- `steps_vs_cfg.png` - Combined parameter correlation
- `sampler_stats.json` - Sampler statistics
- `scheduler_stats.json` - Scheduler statistics
- `term_correlations.json` - Term analysis with correlations
- `report.md` - This report

---

*Generated with Parameter Analysis Module*
"""

        with open(self.output_dir / "report.md", "w") as f:
            f.write(report)

        print(f"  ✓ Report saved to {self.output_dir / 'report.md'}")


def main():
    """Standalone script to run parameter analysis."""
    from shared.io import load_single_jsonl

    print("Parameter Analysis - Standalone Mode")
    print("=" * 50)

    # Load data
    print("Loading data...")
    try:
        vectors_data = load_single_jsonl("output/vectors.jsonl")
        text_data = load_single_jsonl("output/text_data.jsonl")

        if not vectors_data:
            print("✗ No vectors data found. Run data preparation first.")
            return

        print(f"✓ Loaded {len(vectors_data)} vector entries")
        print(f"✓ Loaded {len(text_data)} text entries")

        # Run analysis
        analyzer = ParameterAnalyzer(vectors_data, text_data)
        analyzer.analyze_all()

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
