import argparse
import sys
import os
from pathlib import Path
from typing import Any, Literal

# Ensure package imports resolve when running this file as a script per the project instructions

if __name__ == "__main__":
    root_path = str(Path(__file__).parents[3])
    print(f"root path: {root_path}")
    sys.path.insert(0, root_path)
    if __package__ is None:
        __package__ = "external_modules.step02prepare.full_data"

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
    text_data_file,
)

print("Loading helpers...")
from shared.helpers import remove_models, remove_vectors


def run_prepare(limit: int = 0) -> dict[str, int]:
    print("Loading vector libraries...")
    from shared.vectors.vectors import VectorList
    from shared.image_analysis import ImageAnalysis
    from .data.processing import (
        check_for_leakage,
    )

    print("Starting image processing...")

    if not os.path.isdir(image_root):
        raise FileNotFoundError(
            f"Configured image_root does not exist or is not a directory: {image_root}"
        )

    print("loading index...")
    index_list = load_single_jsonl(index_file)

    processed_files = {s.split("#", 1)[0] for s in index_list}

    print(f"collecting files in {image_root}...")
    files = list(discover_files(image_root))
    collected_data = collect_valid_files(
        files, processed_files, image_root, limit, max_workers=100, scored_only=True
    )

    if len(collected_data) == 0:
        print("No new valid files found. Exiting.")
        return {"total": len(index_list), "new": 0}

    if limit > 0 and len(collected_data) > limit:
        print(
            f"Collected {len(collected_data)} items. Limiting to {limit} as requested."
        )
        collected_data = collected_data[:limit]

    print("analyzing images ...")
    image_analysis = ImageAnalysis(collected_data)
    processed_data = image_analysis.analyze_images_from_paths()
    print("loading existing data...")
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

    print("creating vectors ...")
    vectors_list_parser.create_vectors()
    print("joining vectors...")
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
        print("Dataset updated. Cleaning trained models...")
        remove_models()

    print("=== DONE ===")
    print(f"Total: {summary['total']}, New: {summary['new']}")
    return summary


def run_text_only(limit: int = 0) -> dict[str, int]:
    """
    Process text data only while preserving existing image vectors.
    Useful when text parsing logic changes but images don't need re-analysis.
    """
    print("TEXT-ONLY MODE: Processing text parsing only...")
    from shared.vectors.vectors import VectorList

    if not os.path.isdir(image_root):
        raise FileNotFoundError(
            f"Configured image_root does not exist or is not a directory: {image_root}"
        )

    # Load existing data
    print("loading existing data...")
    index_list = load_single_jsonl(index_file)

    # In text-only mode, we reprocess ALL existing scored images to update their text extraction
    # Build processed_data for VectorList (same format as run_prepare)
    print(f"collecting files in {image_root}...")
    files = list(discover_files(image_root))

    # Get all files that are already scored (match index_list)
    processed_files = {s.split("#", 1)[0] for s in index_list}
    all_collected = collect_valid_files(
        files, set(), image_root, limit=0, max_workers=100, scored_only=True
    )
    if len(all_collected) == 0:
        print("No scored files found. Exiting.")
        return {"total": len(index_list), "new": 0, "text_processed": 0}

    all_collected_dict = {}
    for entry in all_collected:
        img_path: str = entry[0]
        all_collected_dict[img_path] = entry

    # Filter to only the ones we've scored before
    collected_data: list[Any] = []
    last_processed = all_collected[0]

    for item in processed_files:
        if item in all_collected_dict:
            last_processed = item
        else:
            print(
                f"WARNING: file not found during text-only process: {item}, using last found as placeholder"
            )
            last_processed[1]["score"]=-1  # mark as not found

        collected_data.append(last_processed)

    if limit > 0 and len(collected_data) > limit:
        print(
            f"Collected {len(collected_data)} items. Limiting to {limit} as requested."
        )
        collected_data = collected_data[:limit]

    if len(collected_data) == 0:
        print("No scored files found. Exiting.")
        return {"total": len(index_list), "new": 0, "text_processed": 0}

    print(f"Found {len(collected_data)} scored files for text reprocessing")

    # Create VectorList in text-only mode: only processes text data, not images
    vectors_list_parser = VectorList(
        collected_data,
        index_list,
        [],
        [],
        [],
        add_new=False,
        merge_lists=True,
        read_only=False,
        process_images=False,  # CRITICAL: Skip image processing
    )

    print("processing text data only (no image analysis)...")
    try:
        vectors_list_parser.create_vectors()  # This will only process text when process_images=False
        print("joining text data...")
        vectors_list_parser.join_text_data()
        vectors_list_parser.update_lists()

        new_text_list = vectors_list_parser.text_list
        new_score_list = vectors_list_parser.scores_list
        if len(new_text_list) != len(new_score_list) or len(new_text_list) != len(
            index_list
        ):
            raise RuntimeError(
                f"list mismatched lengths: text {len(new_text_list)}, scores {len(new_score_list)}, index {len(index_list)}"
            )

        # Write only text data (preserve vectors)
        print("writing text data...")
        write_single_jsonl(text_data_file, new_text_list, mode="w")
        write_single_jsonl(scores_file, new_score_list, mode="w")

        summary = {
            "total": len(index_list),
            "new": 0,
            "text_processed": len([x for x in new_text_list if x is not None]),
        }

        print(f"TEXT-ONLY PROCESSING COMPLETE")
        print(f"Total: {summary['total']}, Text Processed: {summary['text_processed']}")
        return summary

    except Exception as e:
        print(f"ERROR during text processing: {e}")
        import traceback

        traceback.print_exc()
        raise


def run_rebuild_scores_only() -> dict[str, int]:
    """
    Rebuild the scores file only, preserving the order from the index file.
    For files that still exist, recalculate their scores.
    For files that no longer exist, use the previous cached score.
    """
    print("Starting scores rebuild...")

    # image_root = config["image_root"]
    if not os.path.isdir(image_root):
        raise FileNotFoundError(
            f"Configured image_root does not exist or is not a directory: {image_root}"
        )

    print("Loading existing data...")
    index_list = load_single_jsonl(index_file)
    old_scores_list = load_single_jsonl(scores_file)

    if not index_list:
        print("No index file found. Cannot rebuild scores.")
        return {"total": 0, "updated": 0, "missing": 0}

    print(f"Collecting files in {image_root}...")
    all_files = list(discover_files(image_root))
    collected_data: list[tuple[str, dict[str, Any], str, str]] = collect_valid_files(
        all_files, set([]), image_root, max_workers=40, scored_only=True
    )

    # all data: dict{"file_id":entry}
    final_data: dict[str, dict[str, Any]] = {
        f"{x[3]}#{x[2]}": x[1] for x in collected_data
    }

    # print(list(final_data.keys())[0])
    # print(list(final_data.values())[0])

    new_scores_list: list[float] = []
    updated_count = 0
    missing_count = 0

    for idx, index_entry in enumerate(index_list):
        matching_file = final_data[index_entry] if index_entry in final_data else None

        if matching_file:
            # File exists, recalculate score
            entry = matching_file
            current_score = entry["score"]
            score_modifier = entry.get("score_modifier", 0)
            new_scores_list.append(current_score + (score_modifier * 0.1))
            updated_count += 1
        else:
            # File doesn't exist, use old score
            old_score: Any | Literal[3] = old_scores_list[idx] if idx < len(old_scores_list) else 3
            new_scores_list.append(old_score)
            missing_count += 1
            print(
                f"File not found for index entry: {index_entry}. Using cached score: {old_score}"
            )

    write_single_jsonl(scores_file, new_scores_list, mode="w")

    summary = {
        "total": len(index_list),
        "updated": updated_count,
        "missing": missing_count,
    }
    print("Dataset updated. Cleaning trained models...")
    remove_models()

    print("=== DONE ===")
    print(
        f"Total: {summary['total']}, Updated: {summary['updated']}, Missing: {summary['missing']}"
    )
    return summary


def main(
    rebuild: bool = False,
    test_run: bool = False,
    limit: int = 0,
    steps: bool = False,
    rebuild_scores: bool = False,
    text_only: bool = False,
) -> None:
    print("Starting data prepare...")
    if test_run:
        print("Verifying prepare configuration...")
        print(f"Image root configured as: {config['image_root']}")
        print(f"Vectors file: {config['vectors_file']}")
        print(f"Scores file: {config['scores_file']}")
        print("Test-run finished (no side-effects).")
        return

    if text_only:
        print("TEXT-ONLY MODE: Processing text data only...")
        run_text_only(limit=limit)
        return

    if rebuild:
        print("Rebuild requested: removing existing outputs...")
        remove_vectors()
    if rebuild_scores:
        print("Rebuilding scores file only...")
        run_rebuild_scores_only()
        return
    if limit > 0 and steps:
        print("Steps process enabled")
        new = 1
        i = 0
        while new > 0:
            print(f"step: {i}")
            print("-" * 100)
            summary = run_prepare(limit=limit)
            new = int(summary["new"])
            i += 1
    else:
        run_prepare(limit=limit)


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
        help="Rebuild only the scores file, preserving the index order and reanalyzing files",
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Process text data only, preserving existing image vectors (use when text parsing changes)",
    )
    parser.add_argument(
        "--steps",
        action="store_true",
        help="Process in steps. Must combine with limit, otherwise it will default to full process.",
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
    args = parser.parse_args()
    main(
        rebuild=args.rebuild,
        test_run=args.test_run,
        limit=args.limit,
        steps=args.steps,
        rebuild_scores=args.rebuild_scores,
        text_only=args.text_only,
    )
