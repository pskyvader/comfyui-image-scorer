import argparse
import sys
import os
from pathlib import Path
from typing import Dict, List

# Ensure package imports resolve when running this file as a script per the project instructions
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
print("Importing shared modules...")
from shared.io import atomic_write_json
from shared.config import config
from shared.paths import vectors_file, scores_file, index_file, error_log_file

print("Importing config modules...")
from step02prepare.full_data.config.manager import load_vector_schema

print("Importing data modules...")
from step02prepare.full_data.data.metadata import write_error_log
print("Importing data processing modules...")
from step02prepare.full_data.data.processing import (
    remove_existing_outputs,
    clean_training_artifacts,
    load_existing_data,
    collect_valid_files,
    encode_new_images,
    process_and_append_data,
    check_for_leakage,
)

print("Importing manager modules...")
from step02prepare.full_data.data.manager import collect_files, save_index


def run_prepare(rebuild: bool = False, limit: int = 0) -> Dict[str, int]:
    print("Starting image processing...")

    if rebuild:
        print("Rebuild requested: removing existing outputs...")
        remove_existing_outputs()
        print("files removed")
    index_list, vectors_list, scores_list = load_existing_data()
    processed_files = {s.split("#", 1)[0] for s in index_list}
    error_log: List[dict[str, str]] = []

    image_root = config["image_root"]
    if not os.path.isdir(image_root):
        raise FileNotFoundError(
            f"Configured image_root does not exist or is not a directory: {image_root}"
        )

    files = list(collect_files(image_root))
    collected_data = collect_valid_files(files, processed_files, error_log)

    if limit > 0 and len(collected_data) > limit:
        print(
            f"Collected {len(collected_data)} items. Limiting to {limit} as requested."
        )
        collected_data = collected_data[:limit]
    print("encoding new images ...")
    image_vectors = encode_new_images(collected_data)
    schema = load_vector_schema()
    process_and_append_data(
        collected_data,
        image_vectors,
        vectors_list,
        scores_list,
        index_list,
        processed_files,
        schema,
    )
    check_for_leakage(vectors_list, scores_list)
    atomic_write_json(vectors_file, vectors_list, indent=2)
    atomic_write_json(scores_file, scores_list, indent=2)
    save_index(index_list, index_file)
    write_error_log(error_log, error_log_file)

    summary = {
        "total": len(index_list),
        "new": len(collected_data),
        "errors": len(error_log),
    }

    # Invalidate training cache if data changed
    if summary["new"] > 0:
        print("Dataset updated. Cleaning training artifacts...")
        clean_training_artifacts()

    print("=== DONE ===")
    print(
        f"Total: {summary['total']}, New: {summary['new']}, Errors: {summary['errors']}"
    )
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
