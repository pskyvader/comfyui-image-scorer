from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def _install_vector_import_stubs() -> None:
    stub_specs = {
        "shared.vectors.image_vector": ("ImageVector",),
        "shared.vectors.map_vector": ("MapVector",),
        "shared.vectors.number_vector": ("IntVector", "FloatVector"),
        "shared.vectors.embedding_vector": ("EmbeddingVector",),
    }
    for module_name, class_names in stub_specs.items():
        if module_name in sys.modules:
            continue
        module = types.ModuleType(module_name)
        for class_name in class_names:
            module.__dict__[class_name] = type(class_name, (), {})
        sys.modules[module_name] = module


_install_vector_import_stubs()

from shared.vectors import vectors


class _StubVector:
    def __init__(self, name: str, *args, **kwargs) -> None:
        self.name = name
        self.value_list: list[str] = []
        self.vector_list: list[list[float]] = []
        self.text_list: list[str] = []
        self.path_list: list[str] = []

    def parse_value_list(self, entries, *args, **kwargs):
        self.value_list = []
        for entry in entries:
            value = entry.get(self.name)
            if value is None:
                custom_text = entry.get("custom_text")
                if isinstance(custom_text, dict):
                    value = custom_text.get(self.name)
            if value is None:
                value = ""
            self.value_list.append(str(value))
        return self.value_list

    def create_vector_list(self, *args, **kwargs):
        self.vector_list = [[float(index + 1)] for index, _ in enumerate(self.value_list)]
        return self.vector_list

    def create_text_list(self, *args, **kwargs):
        self.text_list = list(self.value_list)
        return self.text_list

    def create_vector_list_from_paths(self, *args, **kwargs):
        self.vector_list = [[float(index + 1)] for index, _ in enumerate(self.path_list)]
        return self.vector_list


def _patch_vector_stack(monkeypatch) -> None:
    monkeypatch.setattr(vectors, "MapVector", _StubVector)
    monkeypatch.setattr(vectors, "IntVector", _StubVector)
    monkeypatch.setattr(vectors, "FloatVector", _StubVector)
    monkeypatch.setattr(vectors, "EmbeddingVector", _StubVector)
    monkeypatch.setattr(vectors, "ImageVector", _StubVector)
    monkeypatch.setattr(
        vectors,
        "config",
        {
            "vector": {
                "vectors": [
                    {"name": "map_value", "type": "map", "slot_size": 1},
                    {"name": "int_value", "type": "int", "slot_size": 1},
                    {"name": "float_value", "type": "float", "slot_size": 1},
                    {"name": "prompt", "type": "embedding", "slot_size": 1},
                    {"name": "image", "type": "image", "slot_size": 1},
                ]
            }
        },
    )


def test_vector_list_pipeline_and_export(monkeypatch, tmp_path: Path) -> None:
    _patch_vector_stack(monkeypatch)

    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.png"
    image_a.write_bytes(b"")
    image_b.write_bytes(b"")

    raw_data = [
        (
            str(image_a),
            {
                "score": 1.0,
                "map_value": "alpha",
                "int_value": 2,
                "float_value": 3.5,
                "prompt": "hello",
            },
            "ts1",
            "id1",
        ),
        (
            str(image_b),
            {
                "score": 2.0,
                "map_value": "beta",
                "int_value": 4,
                "float_value": 5.5,
                "prompt": "world",
            },
            "ts2",
            "id2",
        ),
    ]

    vector_list = vectors.VectorList(
        raw_data,
        index_list=["existing"],
        vectors_list=[],
        scores_list=[9.0],
        text_list=[{"legacy": True}],
        add_new=True,
        merge_lists=False,
        read_only=False,
        process_images=True,
    )

    assert set(vector_list.sorted_vectors) == {
        "map_value",
        "int_value",
        "float_value",
        "prompt",
        "image",
    }

    vector_list.create_vectors()
    final_vectors = vector_list.join_vectors()
    final_text = vector_list.join_text_data()
    vector_list.update_lists()
    vector_list.split_vectors()
    vector_list.export_split_files(str(tmp_path))

    assert len(final_vectors) == 2
    assert len(final_vectors[0]) == 5
    assert len(final_text) == 2
    assert vector_list.index_list == ["existing", "id1", "id2"]
    assert vector_list.scores_list == [9.0, 1.0, 2.0]
    assert vector_list.vectors_list == final_vectors
    assert vector_list.sorted_vectors["map_value"]["vector"].vector_list == [[1.0], [2.0]]
    assert vector_list.sorted_vectors["image"]["vector"].vector_list == [[1.0], [2.0]]

    split_file = tmp_path / "split" / "map" / "map_value.jsonl"
    assert split_file.exists()
    split_rows = [json.loads(line) for line in split_file.read_text(encoding="utf-8").splitlines()]
    assert split_rows[0]["id"] == "id1"
    assert split_rows[0]["vector"] == [1.0]


def test_vector_list_validate_and_fix_row(monkeypatch) -> None:
    _patch_vector_stack(monkeypatch)

    vector_list = vectors.VectorList(
        [],
        index_list=[],
        vectors_list=[],
        scores_list=[],
        text_list=[],
        add_new=False,
        merge_lists=False,
        read_only=False,
        process_images=False,
    )

    clean = vector_list.validate_and_convert([[1.0], [2.0]], "score", 1)
    assert clean.dtype == np.float32
    assert clean.shape == (2, 1)

    padded = vector_list.fix_row([1.0, 2.0, 3.0, 4.0], expected_total=5)
    assert padded == [1.0, 0.0, 2.0, 3.0, 4.0]
