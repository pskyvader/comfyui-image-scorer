from typing import List, Dict, Any
from .helpers import get_value_from_entry


class IntVector:
    def __init__(self, name: str, max_normalization: int) -> None:
        self.name = name
        self.max_normalization = max_normalization if max_normalization else 10000000
        self.value_list: List[int] = []
        self.vector_list: List[List[int]] = []

    def parse_value_list(self, entries: List[Dict[str, Any]]) -> List[int]:
        for entry in entries:
            # for entry_date in entry.values():
            current_value = get_value_from_entry(entry, self.name)
            if not current_value:
                current_value = 0
            self.value_list.append(current_value)
        return self.value_list

    def create_vector_list(self) -> List[List[int]]:
        for current_value in self.value_list:
            current_value = min(current_value, self.max_normalization)
            self.vector_list.append([current_value])
        return self.vector_list


class FloatVector:
    def __init__(self, name: str, max_normalization: float) -> None:
        self.name = name
        self.max_normalization = max_normalization if max_normalization else 10000000
        self.value_list: List[float] = []
        self.vector_list: List[List[float]] = []

    def parse_value_list(self, entries: List[Dict[str, Any]]) -> List[float]:
        for entry in entries:
            # for entry_date in entry.values():
            current_value = get_value_from_entry(entry, self.name)
            if not current_value:
                current_value = 0
            self.value_list.append(current_value)
        return self.value_list

    def create_vector_list(self) -> List[List[float]]:
        for current_value in self.value_list:
            current_value = min(current_value, self.max_normalization)
            self.vector_list.append([current_value])
        return self.vector_list
