import argparse
import sys
import os
from pathlib import Path
from typing import Dict

# Ensure package imports resolve when running this file as a script per the project instructions

root_path = str(Path(__file__).parent.parent.parent.parent)
sys.path.insert(0, root_path)

from shared.io import (
    load_single_jsonl,
    write_single_jsonl,
    discover_files,
    collect_valid_files,
)

from shared.config import config
from shared.paths import vectors_file, scores_file, index_file
from shared.helpers import remove_models, remove_vectors
from shared.vectors.vectors import VectorList

from shared.image_analysis import ImageAnalysis


from external_modules.step02prepare.full_data.data.processing import (
    check_for_leakage,
)


def run_prepare(rebuild: bool = False, limit: int = 0) -> Dict[str, int]:
    print("Starting image processing...")

    if rebuild:
        print("Rebuild requested: removing existing outputs...")
        remove_vectors()

    image_root = config["image_root"]
    if not os.path.isdir(image_root):
        raise FileNotFoundError(
            f"Configured image_root does not exist or is not a directory: {image_root}"
        )

    print("loading existing data...")
    index_list = load_single_jsonl(index_file)
    vectors_list = load_single_jsonl(vectors_file)
    scores_list = load_single_jsonl(scores_file)

    processed_files = {s.split("#", 1)[0] for s in index_list}

    print(f"collecting files in {image_root}...")
    files = list(discover_files(image_root))
    collected_data = collect_valid_files(
        files, processed_files, image_root, limit, max_workers=20
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
    processed_data = image_analysis.analyze_images_from_paths(
        batch_size=config["prepare"]["vision_model"]["encoding_batch_size"]
    )

    vectors_list_parser = VectorList(
        processed_data,
        index_list,
        vectors_list,
        scores_list,
        add_new=True,
        merge_lists=True,
    )

    print("creating vectors ...")
    vectors_list_parser.create_vectors()
    print("joining vectors...")
    vectors_list_parser.join_vectors()

    new_vectors_list = vectors_list_parser.vectors_list
    new_index_list = vectors_list_parser.index_list
    new_scores_list = vectors_list_parser.scores_list

    check_for_leakage(new_vectors_list, new_scores_list)
    write_single_jsonl(index_file, new_index_list, mode="w")
    write_single_jsonl(vectors_file, new_vectors_list, mode="w")
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


def main(rebuild: bool = False, test_run: bool = False, limit: int = 0) -> None:
    print("Starting data prepare...")
    if test_run:
        print("Verifying prepare configuration...")
        print(f"Image root configured as: {config['image_root']}")
        print(f"Vectors file: {config['vectors_file']}")
        print(f"Scores file: {config['scores_file']}")
        print("Test-run finished (no side-effects).")
        return
    run_prepare(rebuild=rebuild, limit=limit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the prepare pipeline.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Remove existing outputs before processing",
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
    main(rebuild=args.rebuild, test_run=args.test_run, limit=args.limit)
