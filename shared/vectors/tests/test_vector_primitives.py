from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.vectors import embedding_vector, helpers, map_vector, number_vector


def test_l2_normalize_batch_scales_rows() -> None:
    vectors = np.asarray([[3.0, 4.0], [0.0, 5.0]], dtype=np.float32)
    result = helpers.l2_normalize_batch(vectors)

    assert result.shape == (2, 2)
    assert np.allclose(result[0], np.asarray([0.6, 0.8], dtype=np.float32))
    assert np.allclose(result[1], np.asarray([0.0, 1.0], dtype=np.float32))


def test_get_value_from_entry_prefers_alias_then_name() -> None:
    entry = {
        "custom_text": {"prompt": "alias", "caption": "inner"},
        "caption": "outer",
    }

    assert helpers.get_value_from_entry(entry, "caption", ["prompt"]) == "alias"
    assert helpers.get_value_from_entry(entry, "caption", None) == "outer"


def test_int_and_float_vectors_parse_and_clamp() -> None:
    entries = {
        "a": {"score": 12, "ratio": 1.5},
        "b": {"score": 0, "ratio": 3.5},
    }

    int_vector = number_vector.IntVector("score", 10)
    assert int_vector.parse_value_list(entries, None) == {"a": 12, "b": 0}
    assert int_vector.create_vector_list() == {"a": [10], "b": [0]}

    float_vector = number_vector.FloatVector("ratio", 2.0)
    assert float_vector.parse_value_list(entries, None) == {"a": 1.5, "b": 3.5}
    assert float_vector.create_vector_list() == {"a": [1.5], "b": [2.0]}


class _FakeMaps:
    def __init__(self) -> None:
        self.mapping: dict[str, list[str]] = {"color": ["unknown"]}

    def get_value(self, name: str, value: str) -> tuple[int, int]:
        values = self.mapping[name]
        if value in values:
            return values.index(value), len(values)
        return -1, len(values)

    def add_value(self, name: str, value: str) -> tuple[int, int]:
        values = self.mapping[name]
        values.append(value)
        return len(values) - 1, len(values)


def test_map_vector_parse_and_create(monkeypatch) -> None:
    fake_maps = _FakeMaps()
    monkeypatch.setattr(map_vector, "maps_list", fake_maps)
    monkeypatch.setattr(
        map_vector,
        "config",
        {"vector": {"vectors": [{"name": "color", "slot_size": 3}]}},
    )

    vector = map_vector.MapVector("color")
    entries = {
        "a": {"color": "red"},
        "b": {"custom_text": {"color": "blue"}},
    }

    assert vector.parse_value_list(entries, True, None) == {
        "a": {"red": 1.0},
        "b": {"blue": 1.0},
    }
    assert fake_maps.mapping["color"] == ["unknown", "red", "blue"]
    assert vector.create_vector_list() == {
        "a": [0.0, 1.0, 0.0],
        "b": [0.0, 0.0, 1.0],
    }


class _FakeEmbeddingModel:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, batch: list[str]) -> np.ndarray:
        self.calls.append(list(batch))
        rows = []
        for index, _ in enumerate(batch, start=1):
            rows.append([float(index), float(index + 1), 0.0])
        return np.asarray(rows, dtype=np.float32)


def test_embedding_vector_parse_and_encode(monkeypatch) -> None:
    fake_model = _FakeEmbeddingModel()
    monkeypatch.setattr(
        embedding_vector.model_loader,
        "load_embedding_model",
        lambda: (fake_model, 3),
    )

    vector = embedding_vector.EmbeddingVector("prompt", 3)
    entries = {
        "a": {"prompt": "one"},
        "b": {"custom_text": {"prompt": "two"}},
    }

    assert vector.parse_value_list(entries, ["prompt"]) == {
        "a": "one",
        "b": "two",
    }
    assert vector.create_vector_batch(["alpha", "beta"]) == [
        [0.4472135901451111, 0.8944271802902222, 0.0],
        [0.5547001957893372, 0.8320503234863281, 0.0],
    ]
    assert vector.create_vector_list(1) == {
        "a": [0.4472135901451111, 0.8944271802902222, 0.0],
        "b": [0.4472135901451111, 0.8944271802902222, 0.0],
    }
    assert fake_model.calls == [["alpha", "beta"], ["one"], ["two"]]
    assert vector.create_text_batch(["x", "y"]) == ["x", "y"]
    assert vector.create_text_list(1) == ["one", "two"]
