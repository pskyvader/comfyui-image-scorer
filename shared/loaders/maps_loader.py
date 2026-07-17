from typing import Any, cast
import os
import json

from ..paths import maps_dir
from ..logger import get_logger

logger = get_logger(__name__)


# Fixed category vocabularies for per-person / combined maps.
# Order MUST match the model output index order used by attribute_analysis.py
# (FairFace age bins) and the metric names emitted by image_analysis.py.
AGE_CATEGORIES = [
    "0-2",
    "10-19",
    "20-29",
    "3-9",
    "30-39",
    "40-49",
    "50-59",
    "60-69",
    "more than 70",
]
GENDER_CATEGORIES = ["Female", "Male"]
RACE_CATEGORIES = [
    "Black",
    "East Asian",
    "Indian",
    "Latino_Hispanic",
    "Middle Eastern",
    "Southeast Asian",
    "White",
]
ANALYSIS_CATEGORIES = [
    "contrast",
    "sharpness",
    "noise_score",
    "colorfulness",
    "artifact_score",
    "edge_density",
    "texture_lbp",
]


class MapsLoader:
    def __init__(self):
        self.mapping: dict[str,list[str]] = {
            "sampler": [],
            "scheduler": [],
            "model": [],
            "lora": [],
            "age": list(AGE_CATEGORIES),
            "gender": list(GENDER_CATEGORIES),
            "race": list(RACE_CATEGORIES),
            "analysis": list(ANALYSIS_CATEGORIES),
        }

        # Ensure fixed-category map files exist on disk (seeded once).
        for seeded in ("age", "gender", "race", "analysis"):
            file_name = os.path.join(maps_dir, f"{seeded}_map.json")
            if not os.path.exists(file_name):
                self._save_single_map(seeded)

    def get_all_categories(self, name: str) -> list[str]:
        if name not in self.mapping:
            logger.warning(f"Map '{name}' has no registered categories; returning empty list")
            return []
        return list(self.mapping[name])

    def register_value(self, name: str, value: Any) -> None:
        """Idempotently ensure ``value`` (and its sub-keys for dict/list values)
        is present in the map. Unlike :meth:`add_value`, this is safe to call
        repeatedly for the same value: already-present categories are skipped.

        Used while processing text/metadata (and when re-reading split files)
        so the category vocabulary is kept in sync with the data without
        producing duplicates.
        """
        if name not in self.mapping:
            return
        if value is None:
            return
        if isinstance(value, dict):
            for key in value.keys():
                self.register_value(name, key)
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                self.register_value(name, item)
            return
        key = (str(value) or "").strip()
        if key == "" or key == "unknown":
            return
        if key not in self.mapping[name]:
            self.add_value(name, key)

    def add_value(self, name: str, value: str) -> tuple[int, int]:
        """return index of the new added value to the current map

        Args:
            name (str): map name
            value (str): new term

        Raises:
            OverflowError: raise error if max slots limit reached

        Returns:
            tuple[int, int]:
                [0]:  index of the newly added value
                [1]: total length of the map
        """
        current_map = self.mapping[name]
        
        max_slots = 100
        if len(current_map) + 1 > int(max_slots):
            raise OverflowError(f"Map {name} overflowed, max slots {max_slots} reached")

        key = (value or "").strip()
        current_map.append(key)
        self._save_single_map(name)
        return len(current_map) - 1, len(current_map)

    def get_value(self, name: str, value: str) -> tuple[int, int]:
        """get a value from the current map

        Args:
            name (str): map name
            value (str): term to search

        Returns:

            int: index of the term if present, -1 otherwise
        Returns:
            tuple[int, int]:
                [0]: index of the term if present, -1 otherwise
                [1]: total length of the map
        """
        current_map = self.mapping[name]
        key = (value).strip()
        if key == "":
            return 0, len(current_map)
        if key in current_map:
            return current_map.index(key), len(current_map)
        return -1, len(current_map)

    def _save_single_map(self, name: str) -> None:
        current_map = self.mapping[name]
        os.makedirs(maps_dir, exist_ok=True)
        file_name = os.path.join(maps_dir, f"{name}_map.json")
        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(current_map, f, indent=4)

    def _load_single_map(self, name: str) -> list[str]:
        file_name = os.path.join(maps_dir, f"{name}_map.json")
        if os.path.exists(file_name):
            with open(file_name, "r") as f:
                data = json.load(f)
                if not isinstance(data, list):
                    raise TypeError(f"map {name} should be a list")
                return cast(list[str], data)
        empty_map = ["unknown"]
        os.makedirs(maps_dir, exist_ok=True)
        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(empty_map, f, indent=4)
        return empty_map

    def load_maps(self) -> dict[str, list[str]]:
        for key in self.mapping.keys():
            if len(self.mapping[key])==0:
                self.mapping[key]=self._load_single_map(key)
        return self.mapping


maps_list = MapsLoader()
maps_list.load_maps()
