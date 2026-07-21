from __future__ import annotations

import importlib.util
import sys
from collections import deque
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

images_table_stub = ModuleType("external_modules.database_structure.images_table")
images_table_stub.get_all_images = lambda: []  # type: ignore[attr-defined]
sys.modules.setdefault("external_modules.database_structure.images_table", images_table_stub)

comparison_stub = ModuleType("external_modules.comparison")
comparison_stub.__path__ = [str(ROOT / "external_modules" / "comparison")]
algorithm_stub = ModuleType("external_modules.comparison.algorithm")
algorithm_stub.__path__ = [str(ROOT / "external_modules" / "comparison" / "algorithm")]
sys.modules.setdefault("external_modules.comparison", comparison_stub)
sys.modules.setdefault("external_modules.comparison.algorithm", algorithm_stub)

state_path = ROOT / "external_modules" / "comparison" / "algorithm" / "state.py"
state_spec = importlib.util.spec_from_file_location(
    "external_modules.comparison.algorithm.state",
    state_path,
)
assert state_spec and state_spec.loader
state = importlib.util.module_from_spec(state_spec)
sys.modules[state_spec.name] = state
state_spec.loader.exec_module(state)


def test_cached_images_refresh_and_invalidate(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}

    def fake_get_all_images() -> list[dict[str, object]]:
        calls["count"] += 1
        return [{"filename": "a"}, {"filename": "b"}]

    monkeypatch.setattr(state, "get_all_images", fake_get_all_images)
    monkeypatch.setattr(state, "IMAGES_CACHE_TTL", 3600)
    state.invalidate_images_cache()

    first = state.get_cached_all_images()
    second = state.get_cached_all_images()

    assert first == [{"filename": "a"}, {"filename": "b"}]
    assert second == first
    assert calls["count"] == 1

    assert state.get_cached_image("b") == {"filename": "b"}
    assert state.get_cached_image("missing") is None

    state.invalidate_images_cache()
    third = state.get_cached_all_images()
    assert third == first
    assert calls["count"] == 2
