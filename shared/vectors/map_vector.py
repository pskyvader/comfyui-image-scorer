from typing import List, Dict, Any
from ..loaders.maps_loader import maps_list
from ..config import config
from .helpers import get_value_from_entry


class MapVector:
    def __init__(self, name: str) -> None:
        self.name: str = name
        self.value_list: List[str] = []
        self.vector_list: List[List[float]] = []
        self.vector_config = config["vector"]["vectors"]

    def one_hot(self, index: int, length: int) -> List[int]:
        """Return a one-hot encoded vector of ``length`` with 1 at ``index``.

        Out-of-range indexes produce a zero vector rather than raising.
        """
        v = [0] * length
        if 0 <= index < length:
            v[index] = 1
        return v

    def parse_value_list(
        self, entries: List[Dict[str, Any]], add_new_values: bool = False
    ) -> List[str]:
        for entry in entries:
            # for entry_date in entry.values():
            current_value = get_value_from_entry(entry, self.name)
            if not current_value:
                current_value = "unknown"
            index, size = maps_list.get_value(self.name, current_value)
            if index == -1 and add_new_values:
                index, size = maps_list.add_value(self.name, current_value)
                i = next(
                    (
                        index
                        for index, item in enumerate(self.vector_config)
                        if item["name"] == self.name
                    )
                )
                if size > self.vector_config[i]["slot_size"]:
                    self.vector_config[i]["slot_size"] = size
                    config["vector"]["vectors"] = self.vector_config

            self.value_list.append(current_value)
        return self.value_list

    def create_vector_list(self) -> List[List[float]]:
        for current_value in self.value_list:
            index, size = maps_list.get_value(self.name, current_value)
            if index == -1:
                index = 0

            i = next(
                (
                    index
                    for index, item in enumerate(self.vector_config)
                    if item["name"] == self.name
                )
            )

            if size < self.vector_config[i]["slot_size"]:
                size = self.vector_config[i]["slot_size"]

            current_vector = [float(x) for x in self.one_hot(index, size)]
            self.vector_list.append(current_vector)

        return self.vector_list
