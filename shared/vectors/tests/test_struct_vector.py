from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.vectors.struct_vector import StructVector
from shared.vectors.helpers import get_value_from_entry


@pytest.fixture
def base_config() -> dict[str, Any]:
    return {
        "vector": {
            "vectors": [
                {"name": "left_arm", "type": "struct", "slot_size": 12, "per_unit_size": 12},
            ]
        }
    }


def test_init_sets_name() -> None:
    vec = StructVector("left_arm")
    assert vec.name == "left_arm"
    assert vec.value_list == {}
    assert vec.vector_list == {}


def test_parse_value_list_single_person(monkeypatch, base_config) -> None:
    monkeypatch.setattr("shared.vectors.struct_vector.config", base_config)
    monkeypatch.setattr("shared.vectors.struct_vector.get_value_from_entry", get_value_from_entry)
    vec = StructVector("left_arm")
    entries: dict[str, dict[str, list[list[float]]]] = {
        "img1": {"left_arm": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]]},
    }
    result = vec.parse_value_list(entries)
    expected = [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]]
    assert result == {"img1": expected}
    assert len(vec.vector_list) == 0


def test_parse_value_list_resizes_slot(monkeypatch, base_config) -> None:
    monkeypatch.setattr("shared.vectors.struct_vector.config", base_config)
    monkeypatch.setattr("shared.vectors.struct_vector.get_value_from_entry", get_value_from_entry)
    vec = StructVector("left_arm")
    entries = {
        "img1": {"left_arm": [[0.1] * 12, [0.2] * 12]},
    }
    vec.parse_value_list(entries, add_new_values=True)
    cfg = vec._find_config()
    assert cfg is not None
    assert cfg["slot_size"] == 24


def test_create_vector_list_pads_short(monkeypatch, base_config) -> None:
    monkeypatch.setattr("shared.vectors.struct_vector.config", base_config)
    monkeypatch.setattr("shared.vectors.struct_vector.get_value_from_entry", get_value_from_entry)
    vec = StructVector("left_arm")
    vec.value_list = {
        "img1": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]],
    }
    result = vec.create_vector_list()
    assert len(result["img1"]) == 12
    assert result["img1"] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]


def test_create_vector_list_pads_if_empty(monkeypatch, base_config) -> None:
    monkeypatch.setattr("shared.vectors.struct_vector.config", base_config)
    monkeypatch.setattr("shared.vectors.struct_vector.get_value_from_entry", get_value_from_entry)
    vec = StructVector("left_arm")
    vec.value_list = {
        "img1": [],
    }
    result = vec.create_vector_list()
    assert result["img1"] == [0.0] * 12


def test_create_vector_list_truncates_overflow(monkeypatch, base_config) -> None:
    monkeypatch.setattr("shared.vectors.struct_vector.config", base_config)
    monkeypatch.setattr("shared.vectors.struct_vector.get_value_from_entry", get_value_from_entry)
    vec = StructVector("left_arm")
    vec.value_list = {
        "img1": [[1.0] * 20],
    }
    result = vec.create_vector_list()
    assert len(result["img1"]) == 12


def test_multi_image_multi_person(monkeypatch, base_config) -> None:
    monkeypatch.setattr("shared.vectors.struct_vector.config", base_config)
    monkeypatch.setattr("shared.vectors.struct_vector.get_value_from_entry", get_value_from_entry)
    vec = StructVector("left_arm")
    entries = {
        "img1": {"left_arm": [[1.0] * 12]},
        "img2": {"left_arm": [[2.0] * 12, [3.0] * 12]},
    }
    vec.parse_value_list(entries, add_new_values=True)
    vec.create_vector_list()
    assert len(vec.vector_list["img1"]) == 24
    assert vec.vector_list["img1"][:12] == [1.0] * 12
    assert vec.vector_list["img1"][12:] == [0.0] * 12
    assert vec.vector_list["img2"] == [2.0] * 12 + [3.0] * 12


def test_custom_text_value(monkeypatch, base_config) -> None:
    monkeypatch.setattr("shared.vectors.struct_vector.config", base_config)
    monkeypatch.setattr("shared.vectors.struct_vector.get_value_from_entry", get_value_from_entry)
    vec = StructVector("left_arm")
    entries = {
        "img1": {"custom_text": {"left_arm": [[5.0] * 12]}},
    }
    vec.parse_value_list(entries)
    assert vec.value_list["img1"] == [[5.0] * 12]


def test_alias_name(monkeypatch, base_config) -> None:
    custom_config = {
        "vector": {
            "vectors": [
                {"name": "arm_data", "type": "struct", "slot_size": 12, "per_unit_size": 12},
            ]
        }
    }
    monkeypatch.setattr("shared.vectors.struct_vector.config", custom_config)
    monkeypatch.setattr("shared.vectors.struct_vector.get_value_from_entry", get_value_from_entry)
    vec = StructVector("arm_data")
    entries = {
        "img1": {"custom_text": {"arm_data": [[7.0] * 12]}},
    }
    vec.parse_value_list(entries)
    assert vec.value_list["img1"] == [[7.0] * 12]


def test_missing_from_config_raises(monkeypatch) -> None:
    monkeypatch.setattr(
        "shared.vectors.struct_vector.config",
        {"vector": {"vectors": []}},
    )
    vec = StructVector("nonexistent")
    with pytest.raises(KeyError):
        vec.parse_value_list({"img1": {"nonexistent": [[1.0]]}})


def test_hand_landmarks_parsed_correctly(monkeypatch) -> None:
    hand_config = {
        "vector": {
            "vectors": [
                {"name": "left_hand", "type": "struct", "slot_size": 63, "per_unit_size": 63},
            ]
        }
    }
    monkeypatch.setattr("shared.vectors.struct_vector.config", hand_config)
    monkeypatch.setattr("shared.vectors.struct_vector.get_value_from_entry", get_value_from_entry)
    vec = StructVector("left_hand")
    hand_data = [float(i) for i in range(63)]
    entries = {
        "img1": {"left_hand": [hand_data]},
    }
    vec.parse_value_list(entries)
    result = vec.create_vector_list()
    assert result["img1"] == hand_data
