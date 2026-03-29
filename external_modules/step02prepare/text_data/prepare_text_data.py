import argparse
import sys
import os
from typing import Dict, List
from pathlib import Path

# Ensure project root is on sys.path so package imports work when running as a script
if __name__ == "__main__":
    root_path = str(Path(__file__).parents[3])
    sys.path.insert(0, root_path)
    if __package__ is None:
        __package__="external_modules.step02prepare.text_data"




from ..full_data.data.manager import collect_files
from ..full_data.data.metadata import write_error_log
from .text_processing import (
    load_text_index,
    save_text_index,
    process_text_files,
)
from shared.paths import text_data_file,image_root


def run_prepare_text(rebuild: bool = False) -> Dict[str, int]:
    print("Starting text data export...")

    output_file = text_data_file

    if not os.path.isdir(image_root):
        raise FileNotFoundError(
            f"Configured image_root does not exist or is not a directory: {image_root}"
        )
        
        
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
    args = parser.parse_args()
    run_prepare_text(rebuild=args.rebuild)


if __name__ == "__main__":
    main()
