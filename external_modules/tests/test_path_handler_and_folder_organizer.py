from __future__ import annotations

import importlib.util
import json
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


def _prepare_stubs(ranked_root: Path, threshold: int = 1) -> None:
    shared_config_stub = ModuleType("shared.config")
    shared_config_stub.config = {"ranking": {"subfolder_threshold": threshold}}
    sys.modules["shared.config"] = shared_config_stub

    shared_paths_stub = ModuleType("shared.paths")
    shared_paths_stub.image_root_processed = str(ranked_root)  # type: ignore[attr-defined]
    sys.modules["shared.paths"] = shared_paths_stub

    comparisons_stub = ModuleType("external_modules.database_structure.comparisons_table")
    comparisons_stub.get_all_comparisons = lambda: []  # type: ignore[attr-defined]
    sys.modules["external_modules.database_structure.comparisons_table"] = comparisons_stub

    images_stub = ModuleType("external_modules.database_structure.images_table")
    images_stub.get_image = lambda filename: None  # type: ignore[attr-defined]
    sys.modules["external_modules.database_structure.images_table"] = images_stub

    package_stub = ModuleType("external_modules.database_structure")
    package_stub.__path__ = [str(ROOT / "external_modules" / "database_structure")]
    sys.modules["external_modules.database_structure"] = package_stub


def test_compute_path_and_move_json(tmp_path: Path) -> None:
    ranked_root = tmp_path / "ranked"
    _prepare_stubs(ranked_root, threshold=1)
    path_handler = _load_module(
        "external_modules.database_structure.path_handler",
        ROOT / "external_modules" / "database_structure" / "path_handler.py",
    )

    base = ranked_root / "scored_0.8"
    base.mkdir(parents=True, exist_ok=True)
    (base / "existing.png").write_text("x", encoding="utf-8")

    nested = path_handler.compute_path_from_filename("image.png", 0.83)
    assert nested == base / "scored_0.83" / "image.png"

    top_level = path_handler.compute_path_from_filename("image.png", 1.0)
    assert top_level == ranked_root / "scored_1.0" / "image.png"

    current_image = base / "image.png"
    current_json = base / "image.json"
    current_image.write_text("binary", encoding="utf-8")
    current_json.write_text(
        json.dumps({"score": 0.2, "confidence": 0.7, "comparison_history": []}),
        encoding="utf-8",
    )

    path_handler.sync_image_metadata_to_json(
        filename="image.png",
        score=0.83,
        rating_mu=26.0,
        rating_sigma=7.0,
        comparison_count=2,
        all_comparisons=[],
        filename_to_path={"image.png": current_image},
    )

    moved_image = ranked_root / "scored_0.8" / "scored_0.83" / "image.png"
    moved_json = moved_image.with_suffix(".json")
    assert moved_image.exists()
    assert moved_json.exists()
    data = json.loads(moved_json.read_text(encoding="utf-8"))
    assert data["score"] == 0.83
    assert data["rating_mu"] == 26.0
    assert data["rating_sigma"] == 7.0
    assert data["comparison_count"] == 2
    assert "confidence" not in data


def test_folder_organizer_creates_tiers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ranked_root = tmp_path / "ranked"
    _prepare_stubs(ranked_root, threshold=1)
    path_handler = _load_module(
        "external_modules.database_structure.path_handler",
        ROOT / "external_modules" / "database_structure" / "path_handler.py",
    )
    folder_organizer = _load_module(
        "external_modules.database_structure.folder_organizer",
        ROOT / "external_modules" / "database_structure" / "folder_organizer.py",
    )

    monkeypatch.setattr(folder_organizer, "get_ranked_root", lambda: ranked_root)

    assert folder_organizer.ensure_tier_structure() is True
    for i in range(11):
        assert (ranked_root / f"scored_{i / 10.0:.1f}").is_dir()
