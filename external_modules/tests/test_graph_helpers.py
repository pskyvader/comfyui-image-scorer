from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

comparison_stub = ModuleType("external_modules.comparison")
comparison_stub.__path__ = [str(ROOT / "external_modules" / "comparison")]
algorithm_stub = ModuleType("external_modules.comparison.algorithm")
algorithm_stub.__path__ = [str(ROOT / "external_modules" / "comparison" / "algorithm")]
sys.modules.setdefault("external_modules.comparison", comparison_stub)
sys.modules.setdefault("external_modules.comparison.algorithm", algorithm_stub)

graph_helpers_path = ROOT / "external_modules" / "comparison" / "algorithm" / "graph_helpers.py"
graph_helpers_spec = importlib.util.spec_from_file_location(
    "external_modules.comparison.algorithm.graph_helpers",
    graph_helpers_path,
)
assert graph_helpers_spec and graph_helpers_spec.loader
graph_helpers = importlib.util.module_from_spec(graph_helpers_spec)
sys.modules[graph_helpers_spec.name] = graph_helpers
graph_helpers_spec.loader.exec_module(graph_helpers)


def test_topology_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_graph = SimpleNamespace(
        _chain=SimpleNamespace(
            _better_than={"a": set(), "b": {"a"}},
            _worse_than={"a": {"b"}, "b": set()},
            _node_component={"a": 1, "b": 1},
            _component_members={1: ["a", "b"]},
            _chain_length={"a": 2, "b": 2},
        )
    )
    monkeypatch.setattr(graph_helpers, "crystal_graph", fake_graph)

    assert graph_helpers.is_top_node("a") is True
    assert graph_helpers.is_bottom_node("b") is True
    assert graph_helpers.get_node_component("a") == 1
    assert graph_helpers.get_component_members(1) == ["a", "b"]
    assert graph_helpers.get_chain_length("b") == 2


def test_group_nodes_by_extreme(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_graph = SimpleNamespace(
        _chain=SimpleNamespace(
            _better_than={"a": set(), "b": {"a"}, "c": {"x"}},
            _worse_than={"a": {"b"}, "b": set(), "c": set()},
            _node_component={"a": 1, "b": 1, "c": None},
            _component_members={1: ["a", "b"]},
            _chain_length={"a": 1, "b": 1},
        )
    )
    monkeypatch.setattr(graph_helpers, "crystal_graph", fake_graph)

    top_by_component, bottom_by_component = graph_helpers.group_nodes_by_extreme(
        ["a", "b", "c"]
    )

    assert top_by_component == {1: ["a"]}
    assert bottom_by_component == {1: ["b"]}


def test_is_collapsable_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeNode:
        def __init__(self, top: bool, bottom: bool) -> None:
            self._top = top
            self._bottom = bottom

        def is_top(self) -> bool:
            return self._top

        def is_bottom(self) -> bool:
            return self._bottom

    class FakeComponent:
        def __init__(self, comp_id: int) -> None:
            self.id = comp_id

    fake_graph = SimpleNamespace(
        get_node=lambda filename: {
            "a": FakeNode(True, False),
            "b": FakeNode(True, False),
            "c": FakeNode(False, True),
        }.get(filename),
        get_component=lambda node_id=None, **kwargs: FakeComponent(1 if node_id in {"a", "b"} else 2),
        are_in_same_path=lambda a, b: False,
    )
    monkeypatch.setattr(graph_helpers, "crystal_graph", fake_graph)

    assert graph_helpers.is_collapsable_pair("a", "b") is True
    assert graph_helpers.is_collapsable_pair("a", "c") is False
    assert graph_helpers.is_collapsable_pair("a", "missing") is False


def test_filter_excluded_images_and_low_confidence(monkeypatch: pytest.MonkeyPatch) -> None:
    filtered = graph_helpers.filter_excluded_images(
        [{"filename": "a"}, {"filename": "b"}],
        {"b"},
    )
    assert filtered == [{"filename": "a"}]

    assert graph_helpers.find_lowest_confidence_images([]) == []

    images = [
        {"filename": "a", "rating_sigma": 1.0, "score": 0.2},
        {"filename": "b", "rating_sigma": 1.0, "score": 0.1},
        {"filename": "c", "rating_sigma": 0.4, "score": 0.9},
    ]
    monkeypatch.setattr(graph_helpers.random, "random", lambda: 0.0)
    monkeypatch.setattr(graph_helpers.random, "shuffle", lambda values: None)
    result = graph_helpers.find_lowest_confidence_images(images)

    assert [img["filename"] for img in result] == ["b", "a"]
