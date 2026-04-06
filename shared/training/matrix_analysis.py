"""
2D Matrix Analysis for Text Data Parameters

Creates a symmetric matrix combining ALL parameters from text data.
Each row/column is a unique parameter value extracted from text records.
Each cell contains scores for records that have both parameters.
"""

from typing import Any
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import polars as pl

from ..io import atomic_write_json, write_single_jsonl


class MatrixAnalyzer:
    """
    Analyzes ALL parameters by creating a symmetric matrix of parameter values.

    For each record in text_data, extracts all simple parameters (not complex objects).
    Creates rows/columns for each unique value found.
    For each pair of parameters in a record, adds the score to the corresponding cell.
    """

    def __init__(
        self,
        scores: list[float],
        text_data: list[dict[str, Any]],
        memory_limit: int = 10000,
    ):
        """
        Initialize analyzer.

        Args:
            scores: List of score values
            text_data: List of text data dictionaries
            memory_limit: Max unique parameters to track (for memory management)
        """
        self.scores = scores
        self.text_data = text_data
        self.memory_limit = memory_limit

        # All unique parameter values found across all records
        self.all_params: set[str] = set()

        # Mapping of parameter_value -> unique_id (for faster matrix ops)
        self.param_id_map: dict[str, int] = {}
        self.param_list: list[str] = []

        # The symmetric matrix: dict[param1_id][param2_id] = list of scores
        self.matrix: dict[int, dict[int, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

        # Statistics per cell (cached after calculation)
        self.cell_stats: dict[tuple[int, int], dict[str, float]] = {}

    @staticmethod
    def get_text_weight(original_text: str) -> tuple[str, float]:
        """
        Clean up text by removing extra spaces and normalizing.
        Extract weight suffix if present (format: "text:weight"), otherwise return 1.0

        Returns: (normalized_text, weight)
        """
        # Remove parentheses
        original_text = original_text.replace("(", "").replace(")", "").strip()

        # Split by colon to extract weight
        parts = original_text.split(":")
        text = parts[0]
        weight = 1.0

        if len(parts) > 1:
            try:
                weight = float(parts[-1])
            except ValueError:
                weight = 1.0

        # Normalize whitespace and convert to lowercase
        normalized_text = " ".join(str(text).split()).lower()
        return normalized_text, weight

    def _extract_all_params_from_record(self, record: dict[str, Any]) -> list[str]:
        """
        Extract ALL simple parameter values from a record.
        Handles text strings, numbers, model names, samplers, etc.
        Ignores complex objects like nested dicts/lists.

        Special handling:
        - Groups lora + lora_weight together as "lora:lora_name_lora_weight_value"
        - Ensures clip_skip is always represented as negative

        Returns list of string representations of all parameters found.
        """
        params: list[str] = []

        # Handle lora + lora_weight special case: group them together
        lora_value = record.get("lora", None)
        lora_weight = record.get("lora_weight", 0)

        if lora_value is not None and lora_value != "":
            lora_norm, _ = self.get_text_weight(str(lora_value))
            # Round lora_weight to 2 decimals
            lora_weight_rounded = round(
                float(lora_weight) if isinstance(lora_weight, (int, float)) else 0, 2
            )
            param_str = f"lora:{lora_norm}_{lora_weight_rounded}".strip()
            if param_str:
                params.append(param_str)

        # Process all other fields
        for key, value in record.items():
            # Skip lora and lora_weight since we already handled them together
            if key.lower() in ("lora", "lora_weight"):
                continue

            # Special handling for clip_skip: ensure it's negative
            if key.lower() == "clip_skip" and value is not None:
                if isinstance(value, (int, float)):
                    value = -abs(float(value))  # Make absolute then negate
                    self._add_param_from_value(key, value, params)
                continue

            self._add_param_from_value(key, value, params)

        return params

    def _add_param_from_value(
        self, key: str, value: Any, params: list[str], prefix: str = ""
    ) -> None:
        """
        Helper: Extract parameters from a single key-value pair.

        Special handling:
        - clip_skip should already be negative (handled in _extract_all_params_from_record)
        - Float values use 2 decimal precision for better information retention
        - Text weights are normalized to 2 decimals
        """
        if value is None or value == "":
            return

        key_norm, weight = self.get_text_weight(key)

        # Handle scalar types
        if isinstance(value, (str, int, float, bool)):
            if isinstance(value, float):
                # Round to 2 decimals for better precision
                value = round(value * weight, 2)
            param_str = f"{prefix}{key_norm}:{value}".strip()
            if param_str:
                params.append(param_str)

        # Handle lists and tuples
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, (str, int, float)):
                    if isinstance(item, float):
                        # Round to 2 decimals for better precision
                        item = round(item * weight, 2)
                    param_str = f"{prefix}{key_norm}:{item}".strip()
                    params.append(param_str)

                elif isinstance(item, (list, tuple)) and len(item) > 0:
                    # Tuple format: (name, weight) or similar
                    name = str(item[0])
                    tuple_weight = item[1] if len(item) > 1 else ""
                    name_norm, weight2 = self.get_text_weight(name)

                    if isinstance(tuple_weight, (int, float)):
                        # Round to 2 decimals for better precision
                        tuple_weight = round(tuple_weight * weight * weight2, 2)

                    param_str = f"{prefix}{key_norm}:{name_norm}_{tuple_weight}".strip()
                    params.append(param_str)

    def build_matrix(self) -> None:
        """
        Build the complete parameter matrix.

        Process algorithm:
        1. Extract all scores from vectors
        2. Extract all parameters from all records
        3. For each record, extract its parameters
        4. For each pair of parameters in that record, add the score to that cell
        5. Matrix is symmetric: matrix[i][j] == matrix[j][i]
        """
        print(f"Processing {len(self.text_data)} text records...")

        # First pass: collect all unique parameters (with progress)
        all_params_set: set[str] = set()
        with tqdm(
            total=len(self.text_data), desc="Extracting parameters", unit=" records"
        ) as pbar:
            for i, record in enumerate(self.text_data):
                # if (i + 1) % 1000 == 0:
                #     print(
                #         f"  Progress: {i+1}/{len(self.text_data)} records processed for parameter extraction"
                #     )

                params = self._extract_all_params_from_record(record)
                all_params_set.update(params)

                # Limit to avoid memory issues
                pbar.update(1)
                if len(all_params_set) > self.memory_limit:
                    print(
                        f"  WARNING: Parameter limit ({self.memory_limit}) reached, truncating..."
                    )
                    all_params_set = set(
                        sorted(list(all_params_set))[: self.memory_limit]
                    )
                    break

        self.all_params = all_params_set
        self.param_list = sorted(list(all_params_set))

        # Create ID mapping
        for idx, param in enumerate(self.param_list):
            self.param_id_map[param] = idx

        print(f"Found {len(self.param_list)} unique parameters")

        # Second pass: build matrix
        print("Building parameter co-occurrence matrix...")
        with tqdm(
            total=len(self.text_data), desc="Building matrix", unit=" records"
        ) as pbar:
            for i, record in enumerate(self.text_data):
                # if (i + 1) % 1000 == 0:
                #     print(
                #         f"  Progress: {i+1}/{len(self.text_data)} records processed for matrix building"
                #     )

                score = self.scores[i] if i < len(self.scores) else 3.0
                params = self._extract_all_params_from_record(record)

                # Filter to known params (in case of truncation)
                params = [p for p in params if p in self.param_id_map]

                # Add score for all pairs of parameters in this record
                for j, p1 in enumerate(params):
                    p1_id = self.param_id_map[p1]

                    for p2 in params[
                        j:
                    ]:  # Start from j to avoid duplicates and include self-pairs
                        p2_id = self.param_id_map[p2]

                        # Add to matrix (symmetric: matrix[i][j] = matrix[j][i])
                        self.matrix[p1_id][p2_id].append(score)
                        if p1_id != p2_id:
                            self.matrix[p2_id][p1_id].append(score)
                pbar.update(1)
        print(f"Matrix built: {len(self.param_list)}x{len(self.param_list)} parameters")

    def calculate_statistics(
        self, min_count: int = 100
    ) -> dict[tuple[int, int], dict[str, float]]:
        """
        Calculate comprehensive statistics for all matrix cells with sufficient sample count.

        Args:
            min_count: Minimum number of samples required for a cell to be included.
                      Default 100. Reducing to 50-100 gives more granular analysis of
                      potentially harmful parameters. Note: Lower counts = less reliable statistics.

        Statistics calculated for each cell:
        - count: Number of samples in this cell
        - mean: Average score (primary metric for impact)
        - std: Standard deviation (consistency/reliability)
        - min: Minimum score observed
        - max: Maximum score observed
        - median: Median score (robust to outliers)
        - mode: Most common score
        - q1: First quartile (25th percentile)
        - q3: Third quartile (75th percentile)
        - iqr: Interquartile range (Q3-Q1)
        - cv: Coefficient of variation (std/mean, normalized variability)
        - range: Max - Min spread

        Returns dict mapping (p1_id, p2_id) tuple to stats dictionary.
        """
        flattened_data: list[tuple[int, int, float]] = []
        kept_cells = 0
        dropped_cells = 0

        size = len(self.param_list)
        total_possible_cells = (size * (size + 1)) // 2  # Upper triangle + diagonal

        # 1. Flattening Phase with Progress Bar
        with tqdm(
            total=total_possible_cells, desc="Flattening Matrix", unit="cells"
        ) as pbar:
            for p1_id in range(size):
                for p2_id in range(p1_id, size):
                    scores = self.matrix[p1_id][p2_id]

                    if isinstance(scores, list) and len(scores) >= min_count:
                        # Map each individual score to this cell ID
                        for s in scores:
                            flattened_data.append((p1_id, p2_id, float(s)))
                        kept_cells += 1
                    else:
                        dropped_cells += 1

                    # Update progress bar every 10k cells to avoid slowing down the loop
                    if (p1_id * size + p2_id) % 10000 == 0:
                        pbar.update(10000)

            # Ensure bar finishes at 100%
            pbar.n = total_possible_cells
            pbar.refresh()

        print(
            f"Stats: Kept {kept_cells} cells, Dropped {dropped_cells} cells (Min Count: {min_count})"
        )

        if not flattened_data:
            print("No data met the threshold.")
            return {}

        # 2. Polars Calculation Phase
        print("Calculating Statistics via Polars (Multithreaded)...")
        df = pl.DataFrame(
            flattened_data,
            schema=[("p1", pl.Int32), ("p2", pl.Int32), ("score", pl.Float64)],
        )

        stats_df = (
            df.lazy()
            .group_by(["p1", "p2"])
            .agg(
                [
                    pl.len().alias("count"),
                    pl.col("score").mean().alias("mean"),
                    pl.col("score").std().alias("std"),
                    pl.col("score").min().alias("min"),
                    pl.col("score").max().alias("max"),
                    pl.col("score").median().alias("median"),
                    pl.col("score").mode().first().alias("mode"),
                    pl.col("score").quantile(0.25).alias("q1"),
                    pl.col("score").quantile(0.75).alias("q3"),
                ]
            )
            .collect()
        )

        # 3. Reconstruction Phase
        self.cell_stats = {}
        # Re-wrap in tqdm because 30M rows is still a lot to move into a Python dict
        for row in tqdm(
            stats_df.iter_rows(named=True),
            total=len(stats_df),
            desc="Building Final Dict",
        ):
            p1, p2 = row["p1"], row["p2"]

            # Calculate derived statistics
            q1 = float(row["q1"]) if row["q1"] is not None else 0.0
            q3 = float(row["q3"]) if row["q3"] is not None else 0.0
            iqr = q3 - q1

            mean_val = float(row["mean"])
            std_val = float(row["std"]) if row["std"] is not None else 0.0

            # Coefficient of variation (normalized std): std/mean
            # Avoid division by zero; if mean is 0, use high value to indicate extreme variability
            cv = (std_val / mean_val) if mean_val != 0 else 99.9

            range_val = float(row["max"]) - float(row["min"])

            stats = {
                "count": float(row["count"]),
                "mean": mean_val,
                "std": std_val,
                "min": float(row["min"]),
                "max": float(row["max"]),
                "median": float(row["median"]),
                "mode": float(row["mode"]) if row["mode"] is not None else 0.0,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "cv": cv,
                "range": range_val,
            }

            self.cell_stats[(p1, p2)] = stats
            if p1 != p2:
                self.cell_stats[(p2, p1)] = stats

        return self.cell_stats

    def export_to_json(self, output_path: str) -> None:
        """
        Export cell statistics to JSON using atomic write.

        Format: {"param1|param2": {statistics}}
        where statistics contains: count, mean, std, min, max, median, mode
        """
        # Build cell statistics in parameter name format (not matrix indices)
        export_data: dict[str, dict[str, float]] = {}
        export_data_list: list[dict[str, Any]] = []

        with tqdm(
            total=len(self.cell_stats), desc="Exporting to JSON", unit=" cells"
        ) as pbar:
            for (p1_id, p2_id), stats in self.cell_stats.items():
                # Skip duplicates (due to symmetry) - only export upper triangle
                if p1_id > p2_id:
                    pbar.update(1)
                    continue

                # Build parameter names from indices
                p1_param = (
                    self.param_list[p1_id]
                    if p1_id < len(self.param_list)
                    else str(p1_id)
                )
                p2_param = (
                    self.param_list[p2_id]
                    if p2_id < len(self.param_list)
                    else str(p2_id)
                )

                # Use pipe-separated format for clarity
                param_key: str = f"{p1_param}|{p2_param}"
                export_data_list.append({"parameters": param_key, **stats})
                # export_data[param_key] = stats

                pbar.update(1)

        # Use atomic_write_json from shared.io for safe file operations
        # atomic_write_json(output_path, export_data, indent=4)
        write_single_jsonl(output_path, export_data_list, "w")
        print(f"✓ Exported {len(export_data)} cell statistics to {output_path}")

    def print_top_correlations(self, top_n: int = 20) -> None:
        """Print cells with highest mean scores (strongest correlations)."""
        print(f"\nTop {top_n} strongest parameter correlations (by mean score):")
        print("=" * 80)

        # Sort cells by mean score
        sorted_cells = sorted(
            self.cell_stats.items(), key=lambda x: x[1]["mean"], reverse=True
        )[:top_n]

        for (p1_id, p2_id), stats in sorted_cells:
            p1_param = (
                self.param_list[p1_id] if p1_id < len(self.param_list) else str(p1_id)
            )
            p2_param = (
                self.param_list[p2_id] if p2_id < len(self.param_list) else str(p2_id)
            )

            print(f"{p1_param:40s} + {p2_param:40s}")
            print(
                f"  mean: {stats['mean']:.2f} | std: {stats['std']:.2f} | count: {stats['count']}"
            )

    def get_matrix_size(self) -> tuple[int, int]:
        """Get matrix dimensions."""
        return (len(self.param_list), len(self.param_list))

    def get_matrix_summary(self) -> dict[str, Any]:
        """Get summary statistics about the matrix."""
        all_means = [stats["mean"] for stats in self.cell_stats.values()]
        all_counts = [stats["count"] for stats in self.cell_stats.values()]

        return {
            "total_parameters": len(self.param_list),
            "matrix_cells": len(self.cell_stats),
            "total_score_entries": sum(all_counts),
            "mean_of_means": float(np.mean(all_means)) if all_means else 0.0,
            "loaded_records": len(self.text_data),
            "loaded_vectors": len(self.scores),
        }
