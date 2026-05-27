import os
from typing import Any
from ....shared.io import atomic_write_json, load_single_entry_mapping
import logging
import time
logger = logging.getLogger(__name__)


def write_error_log(list[dict[str, str]], path: str) -> None:
    if error_log:
        atomic_write_json(path, error_log, indent=2)
    elif os.path.exists(path):
        os.remove(path)


def load_metadata_entry(
    _start = time.perf_counter()
    _start = time.perf_counter()
    meta_path: str,
    logger.debug("load_metadata_entry took %.4fs", time.perf_counter() - _start)
) -> tuple[dict[str, Any] | None, str | None, str | None]:
    payload, ts, err = load_single_entry_mapping(meta_path)
    if err:
        if err == "not_found":
            return (None, None, "json_not_found")
        if err.startswith("load_error"):
            print(f"Error loading {meta_path}: {err}")
            return (None, None, "bad_json")
        return (None, None, err)
    return (payload, ts, None)
