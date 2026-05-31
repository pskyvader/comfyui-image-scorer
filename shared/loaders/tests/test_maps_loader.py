from __future__ import annotations

import json
from pathlib import Path

import pytest

from shared.loaders import maps_loader


def _patch_maps_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    maps_dir = tmp_path / "maps"
    monkeypatch.setattr(maps_loader, "maps_dir", str(maps_dir))
    return maps_dir


def test_add_value_persists_and_get_value(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    maps_dir = _patch_maps_dir(monkeypatch, tmp_path)
    loader = maps_loader.MapsLoader()

    index, size = loader.add_value("sampler", "euler")

    assert index == 0
    assert size == 1
    assert loader.get_value("sampler", "euler") == (0, 1)
    assert loader.get_value("sampler", "") == (0, 1)
    assert loader.get_value("sampler", "missing") == (-1, 1)

    saved = json.loads((maps_dir / "sampler_map.json").read_text(encoding="utf-8"))
    assert saved == ["euler"]


def test_add_value_rejects_overflow(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_maps_dir(monkeypatch, tmp_path)
    loader = maps_loader.MapsLoader()
    loader.mapping["sampler"] = ["x"] * 100

    with pytest.raises(OverflowError):
        loader.add_value("sampler", "extra")


def test_load_single_map_creates_default_and_validates_type(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    maps_dir = _patch_maps_dir(monkeypatch, tmp_path)
    loader = maps_loader.MapsLoader()

    loaded = loader._load_single_map("model")
    assert loaded == ["unknown"]
    assert json.loads((maps_dir / "model_map.json").read_text(encoding="utf-8")) == [
        "unknown"
    ]

    (maps_dir / "lora_map.json").write_text("{}", encoding="utf-8")
    with pytest.raises(TypeError):
        loader._load_single_map("lora")


def test_load_maps_populates_all_sections(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_maps_dir(monkeypatch, tmp_path)
    loader = maps_loader.MapsLoader()

    mapping = loader.load_maps()

    assert set(mapping) == {"sampler", "scheduler", "model", "lora"}
    assert all(values == ["unknown"] for values in mapping.values())
