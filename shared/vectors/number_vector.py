from typing import Any
from .helpers import get_value_from_entry


class IntVector:
    def __init__(self, name: str, max_normalization: int) -> None:
        self.name = name
        self.max_normalization = max_normalization if max_normalization else 10000000
        self.value_list: list[int] = []
        self.vector_list: list[list[int]] = []

    def parse_value_list(self, entries: list[dict[str, Any]],alias:list[str]|None=None) -> list[int]:
        for entry in entries:
            # for entry_date in entry.values():
            current_value = get_value_from_entry(entry, self.name,alias)
            if not current_value:
                current_value = 0
            self.value_list.append(current_value)
        return self.value_list

    def create_vector_list(self) -> list[list[int]]:
        for current_value in self.value_list:
            current_value = min(current_value, self.max_normalization)
            self.vector_list.append([current_value])
        return self.vector_list


class FloatVector:
    def __init__(self, name: str, max_normalization: float) -> None:
        self.name = name
        self.max_normalization = max_normalization if max_normalization else 10000000
        self.value_list: list[float] = []
        self.vector_list: list[list[float]] = []

    def parse_value_list(self, entries: list[dict[str, Any]],alias:list[str]|None=None) -> list[float]:
        for entry in entries:
            # for entry_date in entry.values():
            current_value = get_value_from_entry(entry, self.name,alias)
            if not current_value:
                current_value = 0
            self.value_list.append(current_value)
        return self.value_list

    def create_vector_list(self) -> list[list[float]]:
        for current_value in self.value_list:
            current_value = min(current_value, self.max_normalization)
            self.vector_list.append([current_value])
        return self.vector_list
