from typing import Any

from ..config import config
from .helpers import get_value_from_entry


class StructVector:
    def __init__(self, name: str) -> None:
        self.name: str = name
        self.value_list: dict[str, list[list[float]]] = {}
        self.vector_list: dict[str, list[float]] = {}
        self.vector_config = config["vector"]["vectors"]

    def _find_config(self) -> dict[str, Any] | None:
        for item in self.vector_config:
            if item["name"] == self.name:
                return item
        return None

    def parse_value_list(
        self,
        entries: dict[str, dict[str, Any]],
        add_new_values: bool = False,
        alias: list[str] | None = None,
    ) -> dict[str, list[list[float]]]:
        cfg = self._find_config()
        if cfg is None:
            raise KeyError(f"Struct vector '{self.name}' not found in config")

        per_unit = cfg.get("per_unit_size", cfg["slot_size"])
        max_people = 0

        for id, entry in entries.items():
            raw = get_value_from_entry(entry, self.name, alias)
            if not isinstance(raw, list):
                raw = []
            self.value_list[id] = raw
            max_people = max(max_people, len(raw))

        if add_new_values and max_people > 0:
            needed = max_people * per_unit
            if needed > cfg["slot_size"]:
                cfg["slot_size"] = needed
                i = next(
                    (i for i, item in enumerate(self.vector_config) if item["name"] == self.name),
                    None,
                )
                if i is not None:
                    self.vector_config[i]["slot_size"] = needed
                    config["vector"]["vectors"] = self.vector_config

        return self.value_list

    def create_vector_list(self) -> dict[str, list[float]]:
        cfg = self._find_config()
        if cfg is None:
            raise KeyError(f"Struct vector '{self.name}' not found in config")

        target = cfg["slot_size"]

        for id, raw_list in self.value_list.items():
            flat: list[float] = []
            for person_data in raw_list:
                if isinstance(person_data, list):
                    flat.extend(person_data)
            if len(flat) > target:
                flat = flat[:target]
            elif len(flat) < target:
                flat.extend([0.0] * (target - len(flat)))
            self.vector_list[id] = flat

        return self.vector_list
