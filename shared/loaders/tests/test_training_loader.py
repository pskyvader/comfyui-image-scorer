from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

import types

helpers_stub = types.ModuleType("shared.helpers")
helpers_stub.remove_directory = lambda path: None  # type: ignore[attr-defined]
sys.modules.setdefault("shared.helpers", helpers_stub)

from ...loaders import training_loader


def _write_jsonl(path: Path, values: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(value) for value in values), encoding="utf-8")


def test_reset_and_remove_training_models(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    loader = training_loader.TrainingLoader(use_cache=True)
    loader.vectors = np.array([1.0], dtype=np.float32)
    loader.scores = np.array([2.0], dtype=np.float32)
    loader.training_model = object()
    loader.feature_rule = np.array([0], dtype=np.int64)
    loader.comparison_rule = (1, {"a.png": (0.5, 3)})
    loader.interaction_data = (np.array([1.0]), np.array([0]))

    removed: list[Path] = []
    monkeypatch.setattr(training_loader, "models_dir", str(tmp_path / "models"))
    monkeypatch.setattr(
        training_loader,
        "remove_directory",
        lambda path: removed.append(Path(path)),
    )

    loader.remove_training_models()

    assert removed == [Path(tmp_path / "models")]
    assert loader.vectors is None
    assert loader.scores is None
    assert loader.training_model is None
    assert loader.feature_rule is None
    assert loader.comparison_rule is None
    assert loader.interaction_data is None


def test_load_vectors_scores_and_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    vectors_path = tmp_path / "vectors.jsonl"
    scores_path = tmp_path / "scores.jsonl"
    _write_jsonl(
        vectors_path,
        [{"a.png": [1, 2]}, {"b.png": [3, 4]}],
    )
    _write_jsonl(scores_path, [{"a.png": 0.5}, {"b.png": 0.9}])

    monkeypatch.setattr(training_loader, "vectors_file", str(vectors_path))
    monkeypatch.setattr(training_loader, "scores_file", str(scores_path))

    loader = training_loader.TrainingLoader(use_cache=True)
    vectors = loader.load_vectors_array()
    scores = loader.load_scores_array()

    assert vectors.shape == (2, 2)
    assert scores.tolist() == [0.5, 0.9]

    _write_jsonl(vectors_path, [{"a.png": [9, 9]}])
    _write_jsonl(scores_path, [{"a.png": 1.0}])
    assert loader.load_vectors_array().shape == (2, 2)
    assert loader.load_scores_array().tolist() == [0.5, 0.9]


def test_filtered_and_interaction_data_roundtrip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    feature_path = tmp_path / "feature_rule.npz"
    interaction_path = tmp_path / "interaction.npz"
    monkeypatch.setattr(training_loader, "feature_rule", str(feature_path))
    monkeypatch.setattr(training_loader, "interaction_data", str(interaction_path))
    monkeypatch.setattr(training_loader, "models_dir", str(tmp_path / "models"))
    rule_path = tmp_path / "comparison_rule.npz"
    monkeypatch.setattr(training_loader, "comparison_rule", str(rule_path))

    loader = training_loader.TrainingLoader(use_cache=True)
    kept = np.array([0], dtype=np.int64)
    loader.save_feature_rule(kept)
    assert loader.load_feature_rule().tolist() == [0]

    loader.save_comparison_rule(2, {"a.png": (0.5, 5), "b.png": (0.75, 7)})
    assert loader.load_comparison_rule(2) == {"a.png": (0.5, 5), "b.png": (0.75, 7)}
    assert loader.load_comparison_rule(3) is None

    x = np.array([[1.0, 2.0]], dtype=np.float32)
    saved_interaction = loader.save_interaction_data(x, kept)
    assert saved_interaction[0].shape == (1, 2)
    assert loader.load_interaction_data()[1].tolist() == [0]

    assert loader._normalize(np.array(5)) == 5
    array_value = np.array([1, 2], dtype=np.int64)
    assert np.array_equal(loader._normalize(array_value), array_value)


def test_training_model_roundtrip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    model_path = tmp_path / "training_model.npz"
    monkeypatch.setattr(training_loader, "training_model", str(model_path))
    monkeypatch.setattr(training_loader, "models_dir", str(tmp_path / "models"))

    loader = training_loader.TrainingLoader(use_cache=True)
    model = {"name": "demo", "value": 7}

    loader.save_training_model(model, {"metric": np.array([1, 2, 3])})

    diagnostics = loader.load_training_model_diagnostics()
    assert "metric" in diagnostics
    assert diagnostics["metric"].tolist() == [1, 2, 3]

    loaded = loader.load_training_model()
    assert loaded == model
    assert loader.load_training_model() is loaded
