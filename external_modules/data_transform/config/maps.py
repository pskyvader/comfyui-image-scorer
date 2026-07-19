from __future__ import annotations

import os
import time

from shared.io import load_json, atomic_write_json
from shared.logger import get_logger
from shared.paths import maps_dir
from shared.config import ensure_dir
from shared.vectors.helpers import get_value_from_entry
from shared.config import config

logger = get_logger(__name__)


def map_dir_path(name: str, path: str | None = None) -> str:
    _start = time.perf_counter()
    result = path if path is not None else os.path.join(maps_dir, f"{name}_map.json")

    return result


def load_map(name: str, path: str | None = None) -> list[str]:
    _start = time.perf_counter()
    ensure_dir(maps_dir)
    map_path = map_dir_path(name, path)
    if not os.path.exists(map_path):
        atomic_write_json(map_path, ["unknown"], indent=4)
        result = ["unknown"]

        return result

    data, _ = load_json(map_path, expect=list)
    result = data if data else ["unknown"]

    return result


def save_map(name: str, data: list[str], path: str | None = None) -> None:
    _start = time.perf_counter()
    ensure_dir(maps_dir)
    map_path = map_dir_path(name, path)
    atomic_write_json(map_path, data, indent=4)


# Not used anywhere in the project right now. Keep the implementation commented
# out per the todo instructions so the module stays minimal.
# def get_or_add(
#     name: str,
#     value: str,
#     max_slots: int | str | None,
#     path: str | None = None,
# ) -> tuple[int, list[str], str]:
#     if isinstance(max_slots, str) and path is None:
#         path = max_slots
#         max_slots = None
#     if max_slots is None:
#         max_slots = 100
#     items = load_map(name, path=path)
#     key = value.strip()
#     if not key:
#         return 0, items, "existing"
#     if key in items:
#         return items.index(key), items, "existing"
#     if len(items) < int(max_slots):
#         items.append(key)
#         save_map(name, items, path=path)
#         return len(items) - 1, items, "added"
#     return 0, items, "overflow"


def register_map_values(processed_data: list) -> None:
    """Register map categories while processing the text/metadata stage.

    Runs over the freshly processed entries (the "text part"), not during
    vector building, so every categorical value (sampler, scheduler, model,
    lora, analysis, age, gender, race) is added to ``maps_list`` before any
    vector is created. Idempotent: already-known categories are skipped.
    """
    from shared.loaders.maps_loader import maps_list

    map_configs = [
        v for v in config["vector"]["vectors"] if v["type"] in ("map", "person_map")
    ]
    if not map_configs:
        return
    for _path, entry, _cat, _extra in processed_data:
        if not isinstance(entry, dict):
            continue
        for v in map_configs:
            name = v["name"]
            alias = v.get("alias")
            value = get_value_from_entry(entry, name, alias)
            if value is None:
                continue
            maps_list.register_value(name, value)
