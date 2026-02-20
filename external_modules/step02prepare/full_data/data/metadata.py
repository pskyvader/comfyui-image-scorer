import os
from typing import Any, Dict, List, Tuple
from shared.io import atomic_write_json, load_single_entry_mapping


def write_error_log(error_log: List[Dict[str, str]], path: str) -> None:
    if error_log:
        atomic_write_json(path, error_log, indent=2)
    elif os.path.exists(path):
        os.remove(path)


def load_metadata_entry(meta_path: str) -> Tuple[Dict[str, Any] | None, str | None, str | None]:
    payload, ts, err = load_single_entry_mapping(meta_path)
    if err:
        if err == "not_found":
            return (None, None, "json_not_found")
        if err.startswith("load_error"):
            print(f"Error loading {meta_path}: {err}")
            return (None, None, "bad_json")
        return (None, None, err)
    return (payload, ts, None)

