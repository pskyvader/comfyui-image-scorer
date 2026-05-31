from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_module(module_name: str, file_path: Path) -> object:
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _prepare_shared_paths(cache_file: Path) -> None:
    shared_paths_stub = ModuleType("shared.paths")
    shared_paths_stub.cache_file = str(cache_file)  # type: ignore[attr-defined]
    sys.modules["shared.paths"] = shared_paths_stub


def _prepare_database_package() -> None:
    package_stub = ModuleType("external_modules.database_structure")
    package_stub.__path__ = [str(ROOT / "external_modules" / "database_structure")]
    sys.modules["external_modules.database_structure"] = package_stub


def test_schema_initializes_database_and_meta(tmp_path: Path) -> None:
    cache_file = tmp_path / "cache.db"
    _prepare_shared_paths(cache_file)
    _prepare_database_package()

    schema = _load_module(
        "external_modules.database_structure.schema",
        ROOT / "external_modules" / "database_structure" / "schema.py",
    )

    assert schema.get_meta_value("db_version") == "4"
    assert schema.get_meta_value("ranking_generation") == "0"

    with schema.get_db_connection() as conn:
        assert conn.row_factory is sqlite3.Row
        tables = {
            row["name"]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert {"meta", "images", "comparisons"} <= tables

    schema._set_meta_value("custom", "value")
    assert schema.get_meta_value("custom") == "value"


def test_images_table_roundtrip(tmp_path: Path) -> None:
    cache_file = tmp_path / "cache.db"
    _prepare_shared_paths(cache_file)
    _prepare_database_package()

    schema = _load_module(
        "external_modules.database_structure.schema",
        ROOT / "external_modules" / "database_structure" / "schema.py",
    )
    images_table = _load_module(
        "external_modules.database_structure.images_table",
        ROOT / "external_modules" / "database_structure" / "images_table.py",
    )

    assert images_table.add_image("a.png", score=0.7, comparison_count=2, prompt_tags="tag")
    row = images_table.get_image("a.png")
    assert row is not None
    assert row["score"] == 0.7
    assert row["comparison_count"] == 2
    assert row["prompt_tags"] == "tag"

    assert images_table.update_image_tags("a.png", "tag2")
    assert images_table.get_image("a.png")["prompt_tags"] == "tag2"

    assert images_table.update_image_score("a.png", 0.8)
    assert images_table.get_image("a.png")["score"] == 0.8

    assert images_table.update_image_rating_state(
        "a.png",
        score=0.9,
        rating_mu=26.0,
        rating_sigma=7.0,
        comparison_count=3,
        touch_timestamp=False,
        last_compared_at="2024-01-01T00:00:00Z",
    )
    updated = images_table.get_image("a.png")
    assert updated["rating_mu"] == 26.0
    assert updated["rating_sigma"] == 7.0
    assert updated["comparison_count"] == 3
    assert updated["last_compared_at"] == "2024-01-01T00:00:00Z"

    assert images_table.reset_all_image_ratings(score=0.1)
    reset = images_table.get_image("a.png")
    assert reset["score"] == 0.1
    assert reset["rating_mu"] == schema.MU0
    assert reset["rating_sigma"] == schema.SIGMA0
    assert reset["comparison_count"] == 0

    all_images = images_table.get_all_images()
    assert len(all_images) == 1
    assert images_table.get_image_count() == 1

    scored, total = images_table.get_scored_images(limit=10, offset=0)
    assert total == 1
    assert len(scored) == 1

    assert images_table.get_images_by_tier(1) == [{"filename": "a.png", **reset}]

    assert images_table.delete_image("a.png") is True
    assert images_table.get_image("a.png") is None
