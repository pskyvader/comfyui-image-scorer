import argparse
import jsonlines
import sys
import os
from pathlib import Path
from typing import Any, Literal
from tqdm import tqdm
import logging

# Ensure package imports resolve when running this file as a script per the project instructions

if __name__ == "__main__":
    root_path = str(Path(__file__).parents[2])
    print(f"root path: {root_path}")
    sys.path.insert(0, root_path)
    if __package__ is None:
        __package__ = "external_modules.data_transform"

from shared.io import (
    load_single_jsonl,
    write_single_jsonl,
    discover_files,
    collect_valid_files,
)

from shared.config import config
from shared.paths import (
    vectors_file,
    scores_file,
    index_file,
    image_root,
    image_root_processed,
    text_data_file,
)
from shared.helpers import remove_models, remove_vectors
from shared.vectors.vectors import VectorList
from shared.image_analysis import ImageAnalysis
from .data.processing import (
    check_for_leakage,
)
import traceback
import time

# Initialize logger
logger = logging.getLogger(__name__)


def run_prepare(int = 0) -> dict[str, int]:
    logger.info("Loading vector libraries...")

    logger.info("Starting image processing...")

    if not os.path.isdir(image_root):
        raise FileNotFoundError(
            f"Configured image_root does not exist or is not a directory: {image_root}"
        )

    logger.info("loading index...")
    index_list = load_single_jsonl(index_file)

    processed_files = {s.split("#", 1)[0] for s in index_list}

    logger.info(f"collecting files in {image_root}...")
    files = list(discover_files(image_root))
    collected_data = collect_valid_files(
        files, processed_files, image_root, limit, max_workers=100, scored_only=True
    )

    if len(collected_data) == 0:
        logger.info("No new valid files found. Exiting.")
        result = {"total": len(index_list), "new": 0}
        logger.debug("run_prepare took %.4fs", time.perf_counter() - _start)
        return result

    if limit > 0 and len(collected_data) > limit:
        logger.info(
            f"Collected {len(collected_data)} items. Limiting to {limit} as requested."
        )
        collected_data = collected_data[:limit]

    logger.info("analyzing images ...")
    image_analysis = ImageAnalysis(collected_data)
    processed_data = image_analysis.analyze_images_from_paths()
    logger.info("loading existing data...")
    vectors_list = load_single_jsonl(vectors_file)
    text_list = load_single_jsonl(text_data_file)
    scores_list = load_single_jsonl(scores_file)
    vectors_list_parser = VectorList(
        processed_data,
        index_list,
        vectors_list,
        scores_list,
        text_list,
        add_new=True,
        merge_lists=True,
    )

    logger.info("creating vectors ...")
    vectors_list_parser.create_vectors()
    vectors_list_parser.export_split_files(os.path.dirname(vectors_file))
    logger.info("joining vectors...")
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

    summary = {
        "total": len(index_list),
        "new": len(processed_data),
    }

    # Invalidate training cache if data changed
    if summary["new"] > 0:
        logger.info("Dataset updated. Cleaning trained models...")
        # remove_models()

    logger.info("=== DONE ===")
    logger.info(f"Total: {summary['total']}, New: {summary['new']}")
    result = summary
    logger.debug("run_prepare took %.4fs", time.perf_counter() - _start)
    return result


def run_text_only(int = 0) -> dict[str, int]:
    """
    Process text data only while preserving existing image vectors.
    Useful when text parsing logic changes but images don't need re-analysis.
    """
    logger.info("TEXT-ONLY MODE: Processing text parsing only...")

    if not os.path.isdir(image_root):
        raise FileNotFoundError(
            f"Configured image_root does not exist or is not a directory: {image_root}"
        )

    # Load existing data
    logger.info("loading existing data...")
    index_list = load_single_jsonl(index_file)
    index_size = len(index_list)
    # In text-only mode, we reprocess ALL existing scored images to update their text extraction
    # Build processed_data for VectorList (same format as run_prepare)
    logger.info(f"collecting files in {image_root}...")
    files = list(discover_files(image_root))

    # Get all files that are already scored (match index_list)
    processed_files = {s.split("#", 1)[0] for s in index_list}
    all_collected = collect_valid_files(
        files, set(), image_root, limit=0, max_workers=100, scored_only=True
    )
    if len(all_collected) == 0:
        logger.info("No scored files found. Exiting.")
        result = {"total": len(index_list), "new": 0, "text_processed": 0}
        logger.debug("run_text_only took %.4fs", time.perf_counter() - _start)
        return result

    all_collected_dict = {}
    for entry in all_collected:
        img_path: str = entry[3]
        all_collected_dict[img_path] = entry

    # Filter to only the ones we've scored before
    collected_data: list[Any] = []
    last_processed: tuple[str, dict[str, Any], str, str] = all_collected[0]

    for item in processed_files:
        if item in all_collected_dict:
            last_processed = all_collected_dict[item]
        else:
            logger.warning(
                f"file not found during text-only process: {item}, using last found as placeholder"
            )
            last_processed[1]["score"] = -1  # mark as not found

        collected_data.append(last_processed)

    if limit > 0 and len(collected_data) > limit:
        logger.info(
            f"Collected {len(collected_data)} items. Limiting to {limit} as requested."
        )
        collected_data = collected_data[:limit]

    if len(collected_data) == 0:
        logger.info("No scored files found. Exiting.")
        result = {"total": len(index_list), "new": 0, "text_processed": 0}
        logger.debug("run_text_only took %.4fs", time.perf_counter() - _start)
        return result

    logger.info(f"Found {len(collected_data)} scored files for text reprocessing")
    # logger.debug(f"collected data sample: {collected_data[0]}")
    logger.info(f"index list size: {len(index_list)} items")

    # Create VectorList in text-only mode: only processes text data, not images
    vectors_list_parser = VectorList(
        collected_data,
        index_list,
        [],
        [],
        [],
        add_new=False,
        merge_lists=False,
        read_only=False,
        process_images=False,  # CRITICAL: Skip image processing
    )
    logger.info(f"index list size: {len(index_list)} items")

    logger.info("processing text data only (no image analysis)...")
    try:
        vectors_list_parser.create_vectors()  # This will only process text when process_images=False
        logger.info("joining text data...")
        vectors_list_parser.join_text_data()
        vectors_list_parser.update_lists()

        new_text_list = vectors_list_parser.text_list
        new_score_list = vectors_list_parser.scores_list
        if (
            len(new_text_list) != len(new_score_list)
            or len(new_text_list) != index_size
        ):
            raise RuntimeError(
                f"list mismatched lengths: text {len(new_text_list)}, scores {len(new_score_list)}, index {len(index_list)}"
            )
        # Write only text data (preserve vectors)
        logger.info("writing text data...")
        write_single_jsonl(text_data_file, new_text_list, mode="w")
        write_single_jsonl(scores_file, new_score_list, mode="w")

        summary = {
            "total": len(index_list),
            "new": 0,
            "text_processed": len([x for x in new_text_list if x is not None]),
        }

        logger.info(f"TEXT-ONLY PROCESSING COMPLETE")
        logger.info(
            f"Total: {summary['total']}, Text Processed: {summary['text_processed']}"
        )
        result = summary
        logger.debug("run_text_only took %.4fs", time.perf_counter() - _start)
        return result

    except Exception as e:
        logger.error(f"ERROR during text processing: {e}")

        traceback.print_exc()
        raise


def run_rebuild_scores_only() -> dict[str, int]:
    """
    Rebuild the scores file only, preserving the order from the index file.
    Reads scores from the ranked JSON companion files in the processed tree.
    For files that no longer exist, uses the previous cached score.
    """
    _start = time.perf_counter()
    _start = time.perf_counter()
    logger.info("Starting scores rebuild from ranked tree...")

    ranked_root = image_root_processed
    if not os.path.isdir(ranked_root):
        raise FileNotFoundError(
            f"Ranked image root does not exist or is not a directory: {ranked_root}"
        )

    logger.info("Loading existing data...")
    index_list = load_single_jsonl(index_file)
    old_scores_list = load_single_jsonl(scores_file)

    if not index_list:
        logger.info("No index file found. Cannot rebuild scores.")
        result = {"total": 0, "updated": 0, "missing": 0}
        logger.debug("run_rebuild_scores_only took %.4fs", time.perf_counter() - _start)
        result = result
        logger.debug("run_rebuild_scores_only took %.4fs", time.perf_counter() - _start)
        return result

    logger.info(f"Collecting files in {ranked_root}...")
    all_files = list(discover_files(ranked_root))
    collected_data: list[tuple[str, dict[str, Any], str, str]] = collect_valid_files(
        all_files, set([]), ranked_root, max_workers=40, scored_only=True
    )

    final_data: dict[str, dict[str, Any]] = {x[3]: x[1] for x in collected_data}

    new_scores_list: list[float] = []
    updated_count = 0
    missing_count = 0

    for idx, index_entry in enumerate(index_list):
        matching_file = final_data.get(index_entry)

        if matching_file:
            new_scores_list.append(float(matching_file["score"]))
            updated_count += 1
        else:
            old_score: Any | Literal[3] = (
                old_scores_list[idx] if idx < len(old_scores_list) else 3
            )
            new_scores_list.append(old_score)
            missing_count += 1
            logger.warning(
                f"File not found in ranked tree: {index_entry}. Using cached score: {old_score}"
            )

    write_single_jsonl(scores_file, new_scores_list, mode="w")

    summary = {
        "total": len(index_list),
        "updated": updated_count,
        "missing": missing_count,
    }
    logger.info("Dataset updated. Cleaning trained models...")
    remove_models()

    logger.info("=== DONE ===")
    logger.info(
        f"Total: {summary['total']}, Updated: {summary['updated']}, Missing: {summary['missing']}"
    )
    result = summary
    logger.debug("run_rebuild_scores_only took %.4fs", time.perf_counter() - _start)
    result = result
    logger.debug("run_rebuild_scores_only took %.4fs", time.perf_counter() - _start)
    return result


def run_rebuild_from_splits() -> dict[str, int]:
    """
    Rebuild the main vectors.jsonl and text_data.jsonl files using the small
    split files in the 'split' directory. This is useful if the main files
    were corrupted or incomplete.
    """
    _start = time.perf_counter()
    _start = time.perf_counter()
    logger.info("Starting rebuild from splits...")

    # Load index file to get the master list of IDs in the correct order
    index_list = load_single_jsonl(index_file)
    if not index_list:
        logger.info("No index file found. Cannot rebuild.")
        result = {"total": 0, "processed": 0}
        logger.debug("run_rebuild_from_splits took %.4fs", time.perf_counter() - _start)
        result = result
        logger.debug("run_rebuild_from_splits took %.4fs", time.perf_counter() - _start)
        return result

    logger.info(f"Index has {len(index_list)} entries.")

    # Get vector config to know which fields to expect
    vector_config = config["vector"]["vectors"]
    vectors_dir_local = os.path.dirname(vectors_file)

    # Data structures to hold loaded split data
    # mapping: field_name -> {id: {"raw": ..., "vector": ...}}
    split_data_map = {}

    for current_type in vector_config:
        v_type = current_type["type"]
        name = current_type["name"]

        split_file = os.path.join(vectors_dir_local, "split", v_type, f"{name}.jsonl")
        if not os.path.exists(split_file):
            print(f"Warning: Split file not found for {name}: {split_file}")
            split_data_map[name] = {}
            continue

        print(f"Loading split file: {split_file}")
        field_data = {}
        # We use jsonlines directly to avoid loading everything at once if we can,
        # though we still store it in field_data dict which will take memory.
        # But this is necessary to reorder according to index_list.
        with jsonlines.open(split_file, mode="r") as reader:
            for obj in reader:
                field_data[obj["id"]] = obj
        split_data_map[name] = field_data

    final_vectors = []
    final_text_data = []

    with tqdm(total=len(index_list), desc="Assembling", unit=" entries") as pbar:
        for uid in index_list:
            row_vector = []
            row_text = {}

            for current_type in vector_config:
                name = current_type["name"]
                v_type = current_type["type"]
                slot_size = current_type["slot_size"]

                entry = split_data_map[name].get(uid)

                if entry:
                    vec = entry.get("vector")
                    raw = entry.get("raw")
                else:
                    vec = None
                    raw = None

                # Handle missing vector (padding with zeros)
                if vec is None:
                    vec = [0.0] * slot_size
                elif len(vec) != slot_size:
                    if len(vec) < slot_size:
                        vec = vec + [0.0] * (slot_size - len(vec))
                    else:
                        vec = vec[:slot_size]

                # Handle missing raw metadata
                if raw is None:
                    if v_type == "embedding":
                        raw = {}
                    else:
                        raw = ""

                row_vector.extend(vec)
                row_text[name] = raw

            final_vectors.append(row_vector)
            final_text_data.append(row_text)
            pbar.update(1)

    logger.info(
        f"Assembled {len(final_vectors)} vectors and {len(final_text_data)} text entries."
    )

    # Write files
    logger.info(f"Writing {vectors_file}...")
    write_single_jsonl(vectors_file, final_vectors, mode="w")
    logger.info(f"Writing {text_data_file}...")
    write_single_jsonl(text_data_file, final_text_data, mode="w")

    logger.info("=== REBUILD COMPLETE ===")
    result = {"total": len(index_list), "processed": len(final_vectors)}
    logger.debug("run_rebuild_from_splits took %.4fs", time.perf_counter() - _start)
    result = result
    logger.debug("run_rebuild_from_splits took %.4fs", time.perf_counter() - _start)
    return result


def main(
    _start = time.perf_counter()
    _start = time.perf_counter()
    rebuild: bool = False,
    test_run: bool = False,
    limit: int = 0,
    batch: bool = False,
    rebuild_scores: bool = False,
    text_only: bool = False,
    rebuild_from_splits: bool = False,
    debug: bool = False,
    logger.debug("main took %.4fs", time.perf_counter() - _start)
) -> None:
    # Configure global logging based on debug flag
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
        force=True,  # Ensure we override any existing configuration
    )

    logger.info("Starting data prepare...")
    if test_run:
        logger.info("Verifying prepare configuration...")
        logger.info(f"Image root configured as: {config['image_root']}")
        logger.info(f"Vectors file: {config['vectors_file']}")
        logger.info(f"Scores file: {config['scores_file']}")
        logger.info("Test-run finished (no side-effects).")
        return

    if text_only:
        logger.info("TEXT-ONLY MODE: Processing text data only...")
        run_text_only(limit=limit)
        return

    if rebuild:
        logger.info("Rebuild requested: removing existing outputs...")
        remove_vectors()
    if rebuild_scores:
        logger.info("Rebuilding scores file only...")
        run_rebuild_scores_only()
        return
    if rebuild_from_splits:
        logger.info("Rebuilding large files from split shards...")
        run_rebuild_from_splits()
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
            i += 1
        if i > 1:
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
        "--rebuild-from-splits",
        action="store_true",
        help="Rebuild large vectors.jsonl and text_data.jsonl from small split files",
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
        "--debug",
        action="store_true",
        help="Enable debug mode (verbose output)",
    )
    args = parser.parse_args()
    main(
        rebuild=args.rebuild,
        test_run=args.test_run,
        limit=args.limit,
        batch=args.batch,
        rebuild_scores=args.rebuild_scores,
        text_only=args.text_only,
        rebuild_from_splits=args.rebuild_from_splits,
        debug=args.debug,
    )
