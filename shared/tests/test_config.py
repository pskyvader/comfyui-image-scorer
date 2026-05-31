from __future__ import annotations

import json
from pathlib import Path

import pytest

from shared import config as config_module


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_path_helpers_and_raw_config_roundtrip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(config_module, "PROJECT_ROOT", tmp_path)

    relative = config_module._get_config_file("config/example.json")
    absolute = config_module._get_config_file(tmp_path / "absolute.json")
    missing = config_module._load_raw_config(tmp_path / "missing.json")

    assert relative == tmp_path / "config/example.json"
    assert absolute == tmp_path / "absolute.json"
    assert missing == {}

    target = tmp_path / "config" / "saved.json"
    config_module._save_raw_config({"value": 1}, target)
    assert json.loads(target.read_text(encoding="utf-8")) == {"value": 1}

    created = tmp_path / "nested" / "dir"
    config_module.ensure_dir(created)
    assert created.is_dir()


def test_auto_save_dict_behaves_like_a_mutable_mapping() -> None:
    saves: list[dict[str, object]] = []

    def save_callback() -> None:
        saves.append({"count": len(saves) + 1})

    wrapper = config_module.AutoSaveDict({"nested": {"value": 1}, "plain": 2}, save_callback)

    nested = wrapper["nested"]
    assert isinstance(nested, config_module.AutoSaveDict)
    assert nested["value"] == 1
    assert wrapper.get("plain") == 2

    wrapper["extra"] = 3
    del wrapper["plain"]

    assert len(wrapper) == 2
    assert list(wrapper) == ["nested", "extra"]
    assert wrapper.copy() == {"nested": {"value": 1}, "extra": 3}
    assert repr(wrapper) == repr({"nested": {"value": 1}, "extra": 3})
    assert len(saves) == 2


def test_config_root_and_subconfig_roundtrip(tmp_path: Path) -> None:
    root = tmp_path / "root.json"
    prepare = tmp_path / "prepare.json"
    training = tmp_path / "training.json"
    vector = tmp_path / "vector.json"
    ranking = tmp_path / "ranking.json"

    _write_json(
        root,
        {
            "prepare_config": str(prepare),
            "training_config": str(training),
            "vector_config": str(vector),
            "ranking_config": str(ranking),
            "root_value": 1,
            "root_dict": {"nested": 7},
        },
    )
    _write_json(prepare, {"prepare_value": 10})
    _write_json(training, {"training_value": 20})
    _write_json(vector, {"vector_value": 30})
    _write_json(ranking, {"ranking_value": 40})

    cfg = config_module.Config(root)

    assert cfg.get("root_value") == 1
    assert cfg["root_dict"]["nested"] == 7
    assert cfg["prepare"]["prepare_value"] == 10

    cfg["root_value"] = 2
    cfg["prepare_value"] = 11
    del cfg["training_value"]

    assert json.loads(root.read_text(encoding="utf-8"))["root_value"] == 2
    assert json.loads(prepare.read_text(encoding="utf-8"))["prepare_value"] == 11
    assert "training_value" not in json.loads(training.read_text(encoding="utf-8"))

    keys = set(iter(cfg))
    assert {"root_value", "root_dict", "prepare", "training", "vector", "ranking"} <= keys
    assert cfg["prepare"]["prepare_value"] == 11
    assert cfg["root_dict"].copy() == {"nested": 7}

    cfg.clear()
    assert cfg["root_value"] == 2

    with pytest.raises(KeyError):
        _ = cfg["missing"]

    cfg["new_root_value"] = 5
    assert json.loads(root.read_text(encoding="utf-8"))["new_root_value"] == 5

    with pytest.raises(KeyError):
        del cfg["does_not_exist"]
