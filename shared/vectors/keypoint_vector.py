from typing import Any

from ..config import config
from .helpers import get_value_from_entry


class KeypointVector:
    """Per-instance keypoint vector (e.g. body pose landmarks).

    Entry value is a list of ``{x, y, z, visibility}`` dicts.
    Slot is ``4 * max_instances`` and grows automatically with more instances.
    """

    PER_UNIT = 4
    KEYS = ("x", "y", "z", "visibility")

    def __init__(self, name: str) -> None:
        self.name: str = name
        self.value_list: dict[str, list[dict[str, float]]] = {}
        self.vector_list: dict[str, list[float]] = {}
        self.vector_config = config["vector"]["vectors"]

    def _config_index(self) -> int:
        return next(
            i for i, item in enumerate(self.vector_config) if item["name"] == self.name
        )

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
        max_instances = 0
        for id, entry in list(entries.items()):
            raw = get_value_from_entry(entry, self.name, alias)
            if not isinstance(raw, list):
                raw = []
            self.value_list[id] = raw
            max_instances = max(max_instances, len(raw))
        if add_new_values and max_instances > 0:
            self._grow(max_instances * self.PER_UNIT)
        return self.value_list

    def create_vector_list(self) -> dict[str, list[float]]:
        target = self.vector_config[self._config_index()]["slot_size"]
        for id, raw in self.value_list.items():
            flat: list[float] = []
            for inst in raw:
                if isinstance(inst, dict):
                    flat.extend(float(inst.get(k, 0.0)) for k in self.KEYS)
                else:
                    flat.extend([0.0] * self.PER_UNIT)
            if len(flat) > target:
                raise ValueError(
                    f"Keypoint vector '{self.name}' for '{id}' has length {len(flat)} "
                    f"but configured slot_size is {target} (more instances than allocated slot)"
                )
            if len(flat) < target:
                flat = flat + [0.0] * (target - len(flat))
            self.vector_list[id] = flat
        return self.vector_list
