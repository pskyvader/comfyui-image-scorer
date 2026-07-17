from typing import Any

from ..loaders.maps_loader import maps_list
from ..config import config
from .helpers import get_value_from_entry


class PersonMapVector:
    """Per-person categorical map (e.g. age / gender / race).

    The entry value is a list of per-person dicts ``[{category: prob}, ...]``.
    The category vocabulary is fixed (seeded in maps_loader) so the slot is
    ``num_categories * max_people`` and grows automatically when a new person
    appears. Each person contributes one contiguous block of softmax weights.
    """

    def __init__(self, name: str) -> None:
        self.name: str = name
        self.value_list: dict[str, list[dict[str, float]]] = {}
        self.vector_list: dict[str, list[float]] = {}
        self.vector_config = config["vector"]["vectors"]

    def _config_index(self) -> int:
        return next(
            i for i, item in enumerate(self.vector_config) if item["name"] == self.name
        )

    def _per_unit(self) -> int:
        return len(maps_list.get_all_categories(self.name))

    def _grow(self, needed: int) -> None:
        i = self._config_index()
        if needed > self.vector_config[i]["slot_size"]:
            self.vector_config[i]["slot_size"] = needed
            config["vector"]["vectors"] = self.vector_config

    def parse_value_list(
        self,
        entries: dict[str, dict[str, Any]],
        add_new_values: bool,
        alias: list[str] | None,
    ) -> dict[str, list[dict[str, float]]]:
        per_unit = self._per_unit()
        max_instances = 0
        for id, entry in list(entries.items()):
            raw = get_value_from_entry(entry, self.name, alias)
            if not isinstance(raw, list):
                raw = []
            for person in raw:
                if isinstance(person, dict):
                    for cat in person:
                        if cat == "":
                            continue
                        index, _ = maps_list.get_value(self.name, cat)
                        if index == -1 and add_new_values:
                            maps_list.add_value(self.name, cat)
            self.value_list[id] = raw
            max_instances = max(max_instances, len(raw))
        if add_new_values and max_instances > 0:
            self._grow(max_instances * per_unit)
        return self.value_list

    def create_vector_list(self) -> dict[str, list[float]]:
        cats = maps_list.get_all_categories(self.name)
        per_unit = len(cats)
        target = self.vector_config[self._config_index()]["slot_size"]
        for id, raw in self.value_list.items():
            flat: list[float] = []
            for person in raw:
                if isinstance(person, dict):
                    flat.extend(float(person.get(cat, 0.0)) for cat in cats)
                else:
                    flat.extend([0.0] * per_unit)
            if len(flat) > target:
                raise ValueError(
                    f"Person-map vector '{self.name}' for '{id}' has length {len(flat)} "
                    f"but configured slot_size is {target} (more instances than allocated slot)"
                )
            if len(flat) < target:
                flat = flat + [0.0] * (target - len(flat))
            self.vector_list[id] = flat
        return self.vector_list
