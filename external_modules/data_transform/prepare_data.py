import argparse
import sys
import os

from pathlib import Path
import time

# import random
from typing import Any, Iterator

if __name__ == "__main__":

    custom_nodes_path = str(Path(__file__).parents[3])

    if custom_nodes_path not in sys.path:
        sys.path.insert(0, custom_nodes_path)

    root_path = str(Path(__file__).parents[2])
    print(f"root path: {root_path}")
    sys.path.insert(0, root_path)
    if __package__ is None:
        __package__ = "comfyui_image_scorer.external_modules.data_transform"

from ...shared.vectors.vectors import VectorList

from ...shared.io import (
    load_single_jsonl,
    write_single_jsonl,
    discover_files,
    collect_valid_files,
)

from ...shared.config import config
from ...shared.paths import (
    vectors_file,
    scores_file,
    comparisons_file,
    index_file,
    image_root,
    text_data_file,
    vectors_data,
    scores_data,
    feature_rule,
    comparison_rule,
    training_model,
)
from ...shared.helpers import remove_derived_caches
from ...shared.analysis.image_analysis import ImageAnalysis
from .config.maps import register_map_values


def _current_scores_dict() -> dict[str, float]:
    """Read the existing scores.jsonl on disk into a filename-keyed dict.

    Returns empty dict if the file is missing, so a first-time rebuild always
    reports a difference.
    """
    out: dict[str, float] = {}
    for rec in load_single_jsonl(scores_file):
        if isinstance(rec, dict) and len(rec) == 1:
            fid, score = next(iter(rec.items()))
            out[str(fid)] = float(score)
    return out


def _current_comparison_counts() -> dict[str, int]:
    """Read the existing comparisons.jsonl and count rows per filename."""
    out: dict[str, int] = {}
    for rec in load_single_jsonl(comparisons_file):
        for key in ("filename_a", "filename_b"):
            fid = rec.get(key)
            if fid:
                out[fid] = out.get(fid, 0) + 1
    return out


def _comparison_counts_from_rows(rows: list[dict[str, str]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for rec in rows:
        for key in ("filename_a", "filename_b"):
            fid = rec.get(key)
            if fid:
                out[fid] = out.get(fid, 0) + 1
    return out


def _scores_changed(new_scores: dict[str, float]) -> bool:
    old = _current_scores_dict()
    if set(old) != set(new_scores):
        return True
    return any(old[k] != new_scores[k] for k in new_scores)


def _comparisons_changed(new_rows: list[dict[str, str]]) -> bool:
    new_counts = _comparison_counts_from_rows(new_rows)
    old = _current_comparison_counts()
    if set(old) != set(new_counts):
        return True
    return any(old.get(k, 0) != new_counts[k] for k in new_counts)


from .data.processing import check_for_leakage
from ..comparison.algorithm.trueskill_rating import (
    public_score_from_rating,
    Rating,
)
from ..comparison.algorithm.comparison_recorder import get_all_comparisons
from ..database_structure.images_table import get_image


from ...shared.logger import configure_package_logging, get_logger, ModuleLogger

logger: ModuleLogger = get_logger(__name__)


def build_comparison_rows(
    index_list: list[str], score_lookup: dict[str, float]
) -> list[dict[str, object]]:
    index_lookup = {filename: idx for idx, filename in enumerate(index_list)}
    comparison_rows: list[dict[str, object]] = []

    for row in get_all_comparisons():
        filename_a = str(row.get("filename_a", ""))
        filename_b = str(row.get("filename_b", ""))
        winner = str(row.get("winner", ""))

        if (
            filename_a not in index_lookup
            or filename_b not in index_lookup
            or winner not in index_lookup
        ):
            continue

        if winner == filename_a:
            loser = filename_b
            winner_side = "a"
        elif winner == filename_b:
            loser = filename_a
            winner_side = "b"
        else:
            continue

        score_a = score_lookup.get(filename_a)
        score_b = score_lookup.get(filename_b)
        score_diff = (
            abs(float(score_a) - float(score_b))
            if score_a is not None and score_b is not None
            else None
        )
        weight_value = row.get("weight", 1.0)
        weight = 1.0 if weight_value is None else float(weight_value)

        comparison_rows.append(
            {
                "comparison_id": int(row.get("id", 0) or 0),
                "filename_a": filename_a,
                "filename_b": filename_b,
                "winner": winner,
                "loser": loser,
                "winner_side": winner_side,
                "index_a": index_lookup[filename_a],
                "index_b": index_lookup[filename_b],
                "winner_index": index_lookup[winner],
                "loser_index": index_lookup[loser],
                "score_a": float(score_a) if score_a is not None else None,
                "score_b": float(score_b) if score_b is not None else None,
                "score_diff": score_diff,
                "weight": weight,
                "transitive_depth": int(row.get("transitive_depth", 0) or 0),
                "timestamp": str(row.get("timestamp", "")),
            }
        )

    return comparison_rows


def run_prepare(limit: int) -> dict[str, int]:
    logger.info("Starting image processing...")

    if not os.path.isdir(image_root):
        raise FileNotFoundError(
            f"Configured image_root does not exist or is not a directory: {image_root}"
        )
    batch_size = config["prepare"]["batch_size"]
    max_workers = config["prepare"]["max_workers"]

    logger.info("loading already-scored files from scores file...")
    # Use the scores file (keyed by filename, matching text_data.jsonl style)
    # as the source of truth for which files are already processed, instead of
    # index.jsonl. index.jsonl is still written for compatibility but is no
    # longer the dedup source.
    scores_records: Iterator[dict[str, float]] = load_single_jsonl(scores_file)
    processed_files: set[str] = {
        str(fid).split("#", 1)[0]
        for rec in scores_records
        if len(rec) == 1
        for fid in rec.keys()
    }

    logger.info(f"collecting files in {image_root}...")
    files: Iterator[tuple[str, str]] = discover_files(image_root)
    # random.shuffle(files)
    collected_data = collect_valid_files(
        files,
        processed_files,
        image_root,
        limit,
        max_workers=max_workers,
        scored_only=True,
    )

    if len(collected_data) == 0:
        logger.info("No new valid files found. Exiting.")
        result = {"total": len(processed_files), "new": 0}

        return result

    if limit > 0 and len(collected_data) > limit:
        logger.info(
            f"Collected {len(collected_data)} items. Limiting to {limit} as requested."
        )
        collected_data = collected_data[:limit]

    logger.info("analyzing images ...")
    image_analysis = ImageAnalysis(collected_data)
    processed_data = image_analysis.analyze_images_from_paths(batch_size, max_workers)
    register_map_values(processed_data)
    logger.info(f"processed data:{len(processed_data)}. Creating vector list object...")
    # print(f"processed data {processed_data}")
    vectors_list_parser = VectorList(
        processed_data,
        read_only=False,
    )

    vectors_list_parser.create_vectors()
    vectors_list_parser.export_split_files()
    vectors_list_parser.filter_missing_vectors()
    vectors_list_parser.join_vectors()
    vectors_list_parser.join_text_data()
    comparison_rows = vectors_list_parser.join_comparison_data()
    vectors_list_parser.update_lists()

    new_vectors_list = vectors_list_parser.vectors_list
    new_text_list = vectors_list_parser.text_list
    new_index_list = vectors_list_parser.index_list
    new_scores_list = vectors_list_parser.scores_list

    # Leakage check operates on the raw aligned arrays, not the keyed records.
    # Both lists are already in the same (unique_ids) order, but to stay
    # independent of index.jsonl we join by filename key.
    vec_by_id = {fid: vec for v in new_vectors_list for fid, vec in v.items()}
    score_by_id = {fid: score for s in new_scores_list for fid, score in s.items()}
    aligned_ids = [
        fid for fid in new_index_list if fid in vec_by_id and fid in score_by_id
    ]
    if len(aligned_ids) != len(new_index_list):
        raise RuntimeError(
            "check leakage: id sets for vectors and scores do not match index. "
            f"vectors ids: {len(vec_by_id)}, scores ids: {len(score_by_id)}, "
            f"index: {len(new_index_list)}"
        )
    check_for_leakage(
        [vec_by_id[fid] for fid in aligned_ids],
        [score_by_id[fid] for fid in aligned_ids],
    )
    write_single_jsonl(index_file, new_index_list, mode="w")
    write_single_jsonl(vectors_file, new_vectors_list, mode="w")
    write_single_jsonl(text_data_file, new_text_list, mode="w")
    write_single_jsonl(scores_file, new_scores_list, mode="w")
    write_single_jsonl(comparisons_file, comparison_rows, mode="w")

    summary = {"total": len(new_index_list), "new": len(processed_data)}

    if summary["new"] > 0:
        logger.info("Dataset updated. Cleaning trained models...")

    logger.info("=== DONE ===")
    logger.info(f"Total: {summary['total']} ({summary['new']} new)")
    result = summary

    return result


def run_rebuild_scores_only() -> dict[str, int]:
    _start = time.perf_counter()
    logger.info("Starting scores rebuild from database...")

    index_list = list(load_single_jsonl(index_file))
    if not index_list:
        logger.info("No index file found. Cannot rebuild scores.")
        result = {"total": 0, "updated": 0, "missing": 0}
        return result

    new_scores_list: list[dict[str, Any]] = []
    updated_count = 0
    missing_count = 0

    for idx, index_entry in enumerate(index_list):
        row = get_image(index_entry)
        if row is None:
            logger.warning("No DB record for: %s", index_entry)
            new_scores_list.append({index_entry: 0.5})
            missing_count += 1
        else:
            mu = float(row["rating_mu"])
            sigma = float(row["rating_sigma"])
            score = public_score_from_rating(Rating(mu=mu, sigma=sigma))
            new_scores_list.append({index_entry: score})
            updated_count += 1

    score_lookup = {
        fid: float(score) for s in new_scores_list for fid, score in s.items()
    }
    comparison_rows = build_comparison_rows(index_list, score_lookup)

    write_single_jsonl(scores_file, new_scores_list, mode="w")
    write_single_jsonl(comparisons_file, comparison_rows, mode="w")

    summary = {
        "total": len(index_list),
        "updated": updated_count,
        "missing": updated_count,
        "comparisons": len(comparison_rows),
    }
    # Only remove caches whose actual source content changed. Vectors are never
    # touched here, so vectors_data is always kept.
    new_scores = score_lookup
    scores_differ = _scores_changed(new_scores)
    comps_differ = _comparisons_changed(comparison_rows)

    to_remove: list[str] = []
    # scores_data, feature_rule and the trained model all depend on the score
    # target, so they are stale only when scores changed.
    if scores_differ:
        to_remove += [scores_data, feature_rule, training_model]
    # comparison_rule is built from comparisons intersected with scores, so it
    # is stale if either the comparisons or the score key set changed.
    if comps_differ or scores_differ:
        to_remove += [comparison_rule]

    if to_remove:
        logger.info(
            "Scores rebuilt. Removing only caches whose sources changed "
            "(scores=%s, comparisons=%s)...",
            scores_differ,
            comps_differ,
        )
        remove_derived_caches(*to_remove)
    else:
        logger.info(
            "Scores rebuilt. No change detected in scores or comparisons; "
            "no caches removed."
        )

    logger.info("=== DONE ===")
    logger.info(
        f"Total: {summary['total']}, Updated: {summary['updated']}, Missing: {summary['missing']}"
    )
    result = summary

    return result


def main(
    # rebuild: bool,
    test_run: bool,
    limit: int,
    batch: bool,
    rebuild_scores: bool,
    # text_only: bool,
    # rebuild_missing: bool,
    debug: bool,
) -> None:
    configure_package_logging(10 if debug else 20)

    logger.info("Starting data prepare...")
    if test_run:
        logger.info("Verifying prepare configuration...")
        logger.info(f"Image root configured as: {config['image_root']}")
        logger.info(f"Vectors file: {config['vectors_file']}")
        logger.info(f"Scores file: {config['scores_file']}")
        logger.info("Test-run finished (no side-effects).")
        return

    # if rebuild:
    #     logger.info("Rebuild requested: removing existing outputs...")
    #     remove_vectors()
    if rebuild_scores:
        logger.info("Rebuilding scores file only...")
        run_rebuild_scores_only()
        return
    if limit > 0 and batch:
        logger.info("batch process enabled")
        new = 1
        i = 0
        while new > 0:
            logger.info(f"step: {i}")
            logger.info("-" * 100)
            summary = run_prepare(limit=limit)
            new = int(summary["new"])
            if new > 0:
                i += 1
        # Only remove derived caches when at least one step actually changed
        # data. run_prepare rewrites all source jsonl only when new > 0, so a
        # zero-change run leaves every cache valid.
        if i > 0:
            logger.info(
                "Prepare produced changes; removing all source-derived caches."
            )
            remove_derived_caches(
                vectors_data,
                scores_data,
                feature_rule,
                comparison_rule,
                training_model,
            )
    else:
        summary = run_prepare(limit=limit)
        new = int(summary["new"])
        if new > 0:
            logger.info(
                "Prepare produced changes; removing all source-derived caches."
            )
            remove_derived_caches(
                vectors_data,
                scores_data,
                feature_rule,
                comparison_rule,
                training_model,
            )
        else:
            logger.info(
                "Prepare found no new images; no source files rewritten, no "
                "caches removed."
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the prepare pipeline.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Remove existing outputs before processing",
    )
    parser.add_argument(
        "--rebuild-scores",
        action="store_true",
        help="Rebuild only the scores file from ranked JSON companions, preserving the index order",
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Process text data only, preserving existing image vectors (use when text parsing changes)",
    )
    parser.add_argument(
        "--rebuild-missing",
        action="store_true",
        help="Rebuild missing or incomplete vector split files from companion data",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process in batch. Must combine with limit, otherwise it will default to full process.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit the number of new items to process (0 for no limit)",
    )
    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Validate configuration and exit without performing processing",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode (verbose output)"
    )
    args = parser.parse_args()
    main(
        # rebuild=args.rebuild,
        test_run=args.test_run,
        limit=args.limit,
        batch=args.batch,
        rebuild_scores=args.rebuild_scores,
        # text_only=args.text_only,
        # rebuild_missing=args.rebuild_missing,
        debug=args.debug,
    )
