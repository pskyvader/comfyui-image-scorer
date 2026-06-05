import argparse
import sys
import os
from pathlib import Path
import time
import logging
import random

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
    index_file,
    image_root,
    text_data_file,
)
from ...shared.helpers import remove_models, remove_vectors
from ...shared.image_analysis import ImageAnalysis
from .data.processing import check_for_leakage
from ..comparison.algorithm.trueskill_rating import (
    public_score_from_rating,
    Rating,
)
from ..database_structure.images_table import get_image


from ...shared.logger import (
    get_logger,
)

logger = get_logger(__name__)


def run_prepare(limit: int) -> dict[str, int]:
    logger.info("Starting image processing...")

    if not os.path.isdir(image_root):
        raise FileNotFoundError(
            f"Configured image_root does not exist or is not a directory: {image_root}"
        )
    batch_size = config["prepare"]["batch_size"]
    max_workers = config["prepare"]["max_workers"]

    logger.info("loading index...")
    index_list = load_single_jsonl(index_file)

    processed_files = {s.split("#", 1)[0] for s in index_list}

    logger.info(f"collecting files in {image_root}...")
    files = list(discover_files(image_root))
    random.shuffle(files)
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
        result = {"total": len(index_list), "new": 0}

        return result

    if limit > 0 and len(collected_data) > limit:
        logger.info(
            f"Collected {len(collected_data)} items. Limiting to {limit} as requested."
        )
        collected_data = collected_data[:limit]

    logger.info("analyzing images ...")
    image_analysis = ImageAnalysis(collected_data)
    processed_data = image_analysis.analyze_images_from_paths(batch_size, max_workers)
    logger.info("Creating vector list object...")
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
    vectors_list_parser.update_lists()

    new_vectors_list = vectors_list_parser.vectors_list
    new_text_list = vectors_list_parser.text_list
    new_index_list = vectors_list_parser.index_list
    new_scores_list = vectors_list_parser.scores_list

    check_for_leakage(new_vectors_list, new_scores_list)
    write_single_jsonl(index_file, new_index_list, mode="w")
    write_single_jsonl(vectors_file, new_vectors_list, mode="w")
    write_single_jsonl(text_data_file, new_text_list, mode="w")
    write_single_jsonl(scores_file, new_scores_list, mode="w")

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

    index_list = load_single_jsonl(index_file)
    if not index_list:
        logger.info("No index file found. Cannot rebuild scores.")
        result = {"total": 0, "updated": 0, "missing": 0}
        return result

    new_scores_list: list[float] = []
    updated_count = 0
    missing_count = 0

    for idx, index_entry in enumerate(index_list):
        row = get_image(index_entry)
        if row is None:
            logger.warning("No DB record for: %s", index_entry)
            new_scores_list.append(0.0)
            missing_count += 1
        else:
            mu = float(row["rating_mu"])
            sigma = float(row["rating_sigma"])
            score = public_score_from_rating(Rating(mu=mu, sigma=sigma))
            new_scores_list.append(score)
            updated_count += 1

    write_single_jsonl(scores_file, new_scores_list, mode="w")

    summary = {
        "total": len(index_list),
        "updated": updated_count,
        "missing": missing_count,
    }
    logger.info("Scores rebuilt. Cleaning trained models...")
    remove_models()

    logger.info("=== DONE ===")
    logger.info(
        f"Total: {summary['total']}, Updated: {summary['updated']}, Missing: {summary['missing']}"
    )
    result = summary

    return result


def main(
    rebuild: bool,
    test_run: bool,
    limit: int,
    batch: bool,
    rebuild_scores: bool,
    text_only: bool,
    rebuild_missing: bool,
    debug: bool,
) -> None:
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
        # force=True,
    )

    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

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
        if i > 0:
            remove_models()
    else:
        summary = run_prepare(limit=limit)
        new = int(summary["new"])
        if new > 0:
            remove_models()


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
        rebuild=args.rebuild,
        test_run=args.test_run,
        limit=args.limit,
        batch=args.batch,
        rebuild_scores=args.rebuild_scores,
        text_only=args.text_only,
        rebuild_missing=args.rebuild_missing,
        debug=args.debug,
    )
