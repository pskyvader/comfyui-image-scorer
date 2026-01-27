import argparse
import sys
import os
from pathlib import Path
from typing import Dict, List

# Ensure project root is on sys.path so package imports work when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from step02prepare.full_data.data.manager import collect_files
from step02prepare.full_data.data.metadata import write_error_log
from shared.config import config, ensure_dir
from step02prepare.text_data.text_processing import (
    load_text_index,
    save_text_index,
    process_text_files,
)


def run_prepare_text(rebuild: bool = False) -> Dict[str, int]:
    print("Starting text data export...")

    output_file = config["text_data_file"]
    index_file = config["text_index_file"]
    image_root = config["image_root"]
    
    if rebuild:
        # Clear existing outputs
        for fpath in [output_file, index_file, config["text_error_log_file"]]:
            if os.path.exists(fpath):
                print(f"Removing {fpath}")
                os.remove(fpath)

    ensure_dir(os.path.dirname(output_file))
    ensure_dir(os.path.dirname(index_file))
    error_log_file = config["text_error_log_file"]
    ensure_dir(os.path.dirname(error_log_file))

    if not os.path.isdir(image_root):
        raise FileNotFoundError(f"image_root not found: {image_root}")

    processed_files = load_text_index(index_file)
    print(f"Index has {len(processed_files)} already processed entries")
    error_log: List[Dict[str, str]] = []
    files = list(collect_files(image_root))
    with open(output_file, "a", encoding="utf-8") as outf:
        new_count = process_text_files(files, processed_files, outf, error_log)

    save_text_index(processed_files, index_file)
    write_error_log(error_log, error_log_file)
    summary = {
        "total": len(processed_files),
        "new": int(new_count),
        "errors": len(error_log),
    }
    print(
        "=== DONE === Total: {} , New: {}, Errors: {}".format(
            summary["total"], summary["new"], summary["errors"]
        )
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the text data export pipeline.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Remove existing outputs before processing",
    )
    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Verify configuration and exit",
    )
    args = parser.parse_args()

    if args.test_run:
        print("Verifying text prepare configuration...")
        try:
            img_root = config["image_root"]
            print(f"Image root configured as: {img_root}")
            if not os.path.isdir(img_root):
                print("Configured image_root does not exist or is not a directory.")
                sys.exit(1)
            print("Configuration looks good.")
            sys.exit(0)
        except Exception as e:
            print(f"Configuration failed: {e}")
            sys.exit(1)

    run_prepare_text(rebuild=args.rebuild)


if __name__ == "__main__":
    main()
