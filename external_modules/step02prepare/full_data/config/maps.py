from __future__ import annotations
import json
import os
from typing import List, Tuple
from shared.config import ensure_dir
from shared.io import load_json
from shared.paths import maps_dir


def map_dir_path(maps_dir: str, name: str, path: str | None = None) -> str:
    if path is not None:
        return path
    return os.path.join(maps_dir, f"{name}_map.json")


def load_map(name: str, path: str | None = None) -> List[str]:
    ensure_dir(maps_dir)
    map_path = map_dir_path(maps_dir, name, path)
    if not os.path.exists(map_path):
        with open(map_path, "w", encoding="utf-8") as f:
            json.dump(["unknown"], f, indent=4)
        return ["unknown"]
    data, err = load_json(map_path, expect=list, default=["unknown"])
    if err:
        return ["unknown"]
    return data or ["unknown"]


def save_map(name: str, data: List[str], path: str | None = None) -> None:
    ensure_dir(maps_dir)
    map_path = map_dir_path(maps_dir, name, path)
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def get_or_add(
    name: str, value: str, max_slots: int | str | None, path: str | None = None
) -> Tuple[int, List[str], str]:
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
