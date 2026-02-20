from typing import Dict, cast, List, Tuple
import os
import json

from ..paths import maps_dir


class MapsLoader:
    def __init__(self):
        self.mapping: Dict[str,List[str]] = {
            "sampler": [],
            "scheduler": [],
            "model": [],
            "lora": [],
        }

    def add_value(self, name: str, value: str) -> Tuple[int, int]:
        """return index of the new added value to the current map

        Args:
            name (str): map name
            value (str): new term

        Raises:
            OverflowError: raise error if max slots limit reached

        Returns:
            Tuple[int, int]:
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

    def get_value(self, name: str, value: str) -> Tuple[int, int]:
        """get a value from the current map

        Args:
            name (str): map name
            value (str): term to search

        Returns:

            int: index of the term if present, -1 otherwise
        Returns:
            Tuple[int, int]:
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

    def _load_single_map(self, name: str) -> List[str]:
        file_name = os.path.join(maps_dir, f"{name}_map.json")
        if os.path.exists(file_name):
            with open(file_name, "r") as f:
                data = json.load(f)
                if not isinstance(data, list):
                    raise TypeError(f"map {name} should be a list")
                return cast(List[str], data)
        empty_map = ["unknown"]
        os.makedirs(maps_dir, exist_ok=True)
        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(empty_map, f, indent=4)
        return empty_map

    def load_maps(self) -> Dict[str, List[str]]:
        for key in self.mapping.keys():
            if len(self.mapping[key])==0:
                self.mapping[key]=self._load_single_map(key)
        return self.mapping


maps_list = MapsLoader()
maps_list.load_maps()
