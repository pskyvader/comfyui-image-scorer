from __future__ import annotations
import json
import os

from ....shared.config import ensure_dir
from ....shared.io import load_json
from ....shared.paths import maps_dir
import logging
import time
logger = logging.getLogger(__name__)


def map_dir_path(str, name: str, path: str | None = None) -> str:
    if path is not None:
        result = path
        logger.debug("map_dir_path took %.4fs", time.perf_counter() - _start)
        return result
    result = os.path.join(maps_dir, f"{name}_map.json")
    logger.debug("map_dir_path took %.4fs", time.perf_counter() - _start)
    return result


def load_map(str, path: str | None = None) -> list[str]:
    ensure_dir(maps_dir)
    map_path = map_dir_path(maps_dir, name, path)
    if not os.path.exists(map_path):
        with open(map_path, "w", encoding="utf-8") as f:
            json.dump(["unknown"], f, indent=4)
        result = ["unknown"]
        logger.debug("load_map took %.4fs", time.perf_counter() - _start)
        return result
    data, err = load_json(map_path, expect=list, default=["unknown"])
    if err:
        result = ["unknown"]
        logger.debug("load_map took %.4fs", time.perf_counter() - _start)
        return result
    result = data or ["unknown"]
    logger.debug("load_map took %.4fs", time.perf_counter() - _start)
    return result


def save_map(str, data: list[str], path: str | None = None) -> None:
    ensure_dir(maps_dir)
    map_path = map_dir_path(maps_dir, name, path)
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def get_or_add(
    _start = time.perf_counter()
    _start = time.perf_counter()
    name: str, value: str, max_slots: int | str | None, path: str | None = None
    logger.debug("get_or_add took %.4fs", time.perf_counter() - _start)
) -> tuple[int, list[str], str]:
    if isinstance(max_slots, str) and path is None:
        path = max_slots
        max_slots = None
    if max_slots is None:
        max_slots = 100
    mp = load_map(name, path=path)
    key = (value or "").strip()
    if key == "":
        return (0, mp, "existing")
    if key in mp:
        return (mp.index(key), mp, "existing")
    if len(mp) < int(max_slots):
        mp.append(key)
        save_map(name, mp, path=path)
        return (len(mp) - 1, mp, "added")
    return (0, mp, "overflow")
