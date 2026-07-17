from typing import Any

from ..loaders.maps_loader import maps_list
from ..config import config
from .helpers import get_value_from_entry


class MapVector:
    """Categorical map with float weights.

    Accepts several value shapes (all stored as a weighted category dict):
      - ``str``                -> {str: 1.0}
      - ``list[str]``          -> {s: 1.0 for s in list}  (multi-hot, single instance)
      - ``dict[str, float]``   -> weighted single instance (e.g. lora or analysis)
      - ``list[dict]``         -> first dict used (per-person maps use person_map type)

    Categories are discovered through ``maps_list`` so the slot grows automatically
    and the grown ``slot_size`` is persisted to disk via the AutoSave config.
    """

    def __init__(self, name: str) -> None:
        self.name: str = name
        self.value_list: dict[str, dict[str, float]] = {}
        self.vector_list: dict[str, list[float]] = {}
        self.vector_config = config["vector"]["vectors"]

    def _config_index(self) -> int:
        return next(
            i for i, item in enumerate(self.vector_config) if item["name"] == self.name
        )

    def _maybe_grow(self, size: int) -> None:
        i = self._config_index()
        if size > self.vector_config[i]["slot_size"]:
            self.vector_config[i]["slot_size"] = size
            config["vector"]["vectors"] = self.vector_config

    def _normalize(self, value: Any) -> dict[str, float]:
        if value is None:
            return {}
        if isinstance(value, str):
            return {value: 1.0} if value else {}
        if isinstance(value, dict):
            return {str(k): float(v) for k, v in value.items() if k != ""}
        if isinstance(value, list):
            if value and isinstance(value[0], dict):
                first = value[0]
                return {str(k): float(v) for k, v in first.items() if k != ""}
            return {str(v): 1.0 for v in value if v != ""}
        return {}

    def parse_value_list(
        self,
        entries: dict[str, dict[str, Any]],
        add_new_values: bool,
        alias: list[str] | None,
    ) -> dict[str, dict[str, float]]:
        for id, entry in list(entries.items()):
            current_value = get_value_from_entry(entry, self.name, alias)
            norm = self._normalize(current_value)
            for cat in norm:
                index, size = maps_list.get_value(self.name, cat)
                if index == -1:
                    if add_new_values:
                        index, size = maps_list.add_value(self.name, cat)
                    else:
                        continue
                self._maybe_grow(size)
            self.value_list[id] = norm
        return self.value_list

    def create_vector_list(self) -> dict[str, list[float]]:
        cats = maps_list.get_all_categories(self.name)
        size = len(cats)
        target = self.vector_config[self._config_index()]["slot_size"]
        for id, norm in self.value_list.items():
            vec = [0.0] * size
            for cat, weight in norm.items():
                index, _ = maps_list.get_value(self.name, cat)
                if index != -1:
                    vec[index] = weight
            if len(vec) > target:
                raise ValueError(
                    f"Map vector '{self.name}' for '{id}' has length {len(vec)} "
                    f"but configured slot_size is {target} (category vocabulary "
                    f"exceeds the allocated slot)"
                )
            if len(vec) < target:
                vec = vec + [0.0] * (target - len(vec))
            self.vector_list[id] = vec
        return self.vector_list
