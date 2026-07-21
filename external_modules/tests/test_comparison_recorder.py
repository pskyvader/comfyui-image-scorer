from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

images_table_stub = ModuleType("external_modules.database_structure.images_table")
images_table_stub.get_all_images = lambda: []  # type: ignore[attr-defined]
images_table_stub.get_image = lambda filename: None  # type: ignore[attr-defined]
images_table_stub.update_image_rating_state = lambda **kwargs: True  # type: ignore[attr-defined]
sys.modules.setdefault("external_modules.database_structure.images_table", images_table_stub)

comparisons_table_stub = ModuleType("external_modules.database_structure.comparisons_table")
comparisons_table_stub.add_comparison = lambda **kwargs: 1  # type: ignore[attr-defined]
comparisons_table_stub.comparison_exists_for_pair = lambda a, b: False  # type: ignore[attr-defined]
comparisons_table_stub.get_all_comparisons = lambda: []  # type: ignore[attr-defined]
sys.modules.setdefault(
    "external_modules.database_structure.comparisons_table",
    comparisons_table_stub,
)

path_handler_stub = ModuleType("external_modules.database_structure.path_handler")
path_handler_stub.sync_image_metadata_to_json = lambda **kwargs: True  # type: ignore[attr-defined]
sys.modules.setdefault("external_modules.database_structure.path_handler", path_handler_stub)

shared_graph_stub = ModuleType("shared.graph")
shared_graph_stub.__path__ = [str(ROOT / "shared" / "graph")]
shared_graph_crystal_stub = ModuleType("shared.graph.crystal_graph")
shared_graph_crystal_stub.crystal_graph = SimpleNamespace(
    apply_comparison=lambda a, b: None
)
sys.modules.setdefault("shared.graph", shared_graph_stub)
sys.modules.setdefault("shared.graph.crystal_graph", shared_graph_crystal_stub)

comparison_stub = ModuleType("external_modules.comparison")
comparison_stub.__path__ = [str(ROOT / "external_modules" / "comparison")]
algorithm_stub = ModuleType("external_modules.comparison.algorithm")
algorithm_stub.__path__ = [str(ROOT / "external_modules" / "comparison" / "algorithm")]
sys.modules.setdefault("external_modules.comparison", comparison_stub)
sys.modules.setdefault("external_modules.comparison.algorithm", algorithm_stub)

comparison_recorder_path = (
    ROOT / "external_modules" / "comparison" / "algorithm" / "comparison_recorder.py"
)
comparison_recorder_spec = importlib.util.spec_from_file_location(
    "external_modules.comparison.algorithm.comparison_recorder",
    comparison_recorder_path,
)
assert comparison_recorder_spec and comparison_recorder_spec.loader
comparison_recorder = importlib.util.module_from_spec(comparison_recorder_spec)
sys.modules[comparison_recorder_spec.name] = comparison_recorder
comparison_recorder_spec.loader.exec_module(comparison_recorder)


def test_update_scores_after_comparison(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        comparison_recorder,
        "rating_from_row",
        lambda row: SimpleNamespace(mu_skill=float(row["rating_mu"]), sigma_uncertainty=float(row["rating_sigma"])),
    )
    monkeypatch.setattr(
        comparison_recorder,
        "update_ratings",
        lambda winner, loser: (SimpleNamespace(mu_skill=30.0, sigma_uncertainty=4.0), SimpleNamespace(mu_skill=20.0, sigma_uncertainty=6.0)),
    )
    monkeypatch.setattr(comparison_recorder, "public_score_from_rating", lambda rating: rating.mu_skill / 100.0)

    winner, loser = comparison_recorder.update_scores_after_comparison(
        "a",
        "b",
        {"rating_mu": 25.0, "rating_sigma": 8.0, "comparison_count": 1},
        {"rating_mu": 20.0, "rating_sigma": 9.0, "comparison_count": 3},
        1.0,
    )

    assert winner["rating_mu"] == 30.0
    assert loser["rating_sigma"] == 6.0
    assert winner["score"] == 0.3
    assert loser["comparison_count"] == 4


def test_persist_image_state(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_update_image_rating_state(**kwargs):
        captured.update(kwargs)
        return True

    monkeypatch.setattr(
        comparison_recorder,
        "update_image_rating_state",
        fake_update_image_rating_state,
    )

    result = comparison_recorder._persist_image_state(
        "a",
        {"score": 0.8, "rating_mu": 27.0, "rating_sigma": 7.0, "comparison_count": 2},
    )

    assert result is True
    assert captured["filename"] == "a"
    assert captured["touch_timestamp"] is True


def test_record_comparison_success_and_shortcuts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        comparison_recorder,
        "comparison_exists_for_pair",
        lambda a, b: False,
    )
    monkeypatch.setattr(
        comparison_recorder,
        "get_image_data",
        lambda filename: {
            "filename": filename,
            "score": 0.5,
            "rating_mu": 25.0,
            "rating_sigma": 8.0,
            "comparison_count": 1,
        },
    )
    monkeypatch.setattr(
        comparison_recorder,
        "update_scores_after_comparison",
        lambda *args, **kwargs: (
            {
                "filename": "a",
                "score": 0.9,
                "rating_mu": 30.0,
                "rating_sigma": 4.0,
                "comparison_count": 2,
            },
            {
                "filename": "b",
                "score": 0.1,
                "rating_mu": 20.0,
                "rating_sigma": 6.0,
                "comparison_count": 2,
            },
        ),
    )
    monkeypatch.setattr(comparison_recorder, "add_comparison", lambda **kwargs: 42)
    monkeypatch.setattr(comparison_recorder, "get_all_comparisons", lambda: [{"id": 1}])

    persist_calls: list[str] = []
    monkeypatch.setattr(
        comparison_recorder,
        "_persist_image_state",
        lambda filename, data: persist_calls.append(filename) or True,
    )

    sync_calls: list[str] = []

    def fake_sync_image_metadata_to_json(**kwargs):
        sync_calls.append(kwargs["filename"])
        return True

    monkeypatch.setattr(
        comparison_recorder,
        "sync_image_metadata_to_json",
        fake_sync_image_metadata_to_json,
    )

    apply_calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        comparison_recorder,
        "crystal_graph",
        SimpleNamespace(apply_comparison=lambda a, b: apply_calls.append((a, b))),
    )

    invalidate_calls = {"count": 0}
    monkeypatch.setattr(
        comparison_recorder,
        "invalidate_images_cache",
        lambda: invalidate_calls.__setitem__("count", invalidate_calls["count"] + 1),
    )

    assert comparison_recorder.record_comparison("a", "b", "a", 1.0, 2) is True
    assert persist_calls == ["a", "b"]
    assert sync_calls == ["a", "b"]
    assert apply_calls == [("a", "b")]
    assert invalidate_calls["count"] == 1

    monkeypatch.setattr(comparison_recorder, "comparison_exists_for_pair", lambda a, b: True)
    assert comparison_recorder.record_comparison("a", "b", "a", 1.0, 2) is False
