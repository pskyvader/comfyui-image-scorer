from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from .. import helpers, io, utils


def test_jsonl_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    payload = [{"a": 1}, {"b": "two"}]

    io.write_single_jsonl(str(path), payload, mode="w")
    assert io.load_single_jsonl(str(path)) == payload


def test_discover_and_collect_valid_files(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()

    img_a = root / "a.png"
    json_a = root / "a.json"
    img_b = root / "b.jpg"
    json_b = root / "b.json"
    img_a.write_bytes(b"")
    img_b.write_bytes(b"")
    json_a.write_text(
        json.dumps({"score": 1.2, "comparison_count": 6, "payload": "x"}),
        encoding="utf-8",
    )
    json_b.write_text(json.dumps({"score": 2.5}), encoding="utf-8")

    discovered = list(io.discover_files(str(root)))
    assert (str(img_a), str(json_a)) in discovered
    assert (str(img_b), str(json_b)) in discovered

    collected = io.collect_valid_files(discovered, set(), str(root), limit=1, max_workers=2, scored_only=True)
    assert len(collected) == 1
    assert collected[0][0] in {str(img_a), str(img_b)}


def test_collect_single_and_recursive_json(tmp_path: Path) -> None:
    json_path = tmp_path / "meta.json"
    json_path.write_text(
        json.dumps({"wrapped": "{\"nested\": [1, 2, 3]}"}),
        encoding="utf-8",
    )

    parsed, err = io.load_json(str(json_path), expect=dict, default=None)
    assert err is None
    assert parsed["wrapped"]["nested"] == [1, 2, 3]

    single = tmp_path / "single.json"
    single.write_text(json.dumps({"image-1": {"score": 7}}), encoding="utf-8")
    payload, key, err = io.load_single_entry_mapping(str(single))
    assert err is None
    assert key == "image-1"
    assert payload == {"score": 7}


def test_atomic_write_json(tmp_path: Path) -> None:
    path = tmp_path / "out" / "data.json"
    io.atomic_write_json(str(path), {"x": 1}, indent=2)
    assert json.loads(path.read_text(encoding="utf-8")) == {"x": 1}


def test_utils_parse_custom_text_and_first_present() -> None:
    assert utils.parse_custom_text(None) == {}
    assert utils.parse_custom_text("{'a': 1}") == {"a": 1}
    assert utils.first_present({"a": None, "b": 2}, ("a", "b"), default=0) == 2
    assert utils.first_present({}, ("missing",), default=5) == 5


def test_helper_directory_and_export(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    target = tmp_path / "remove_me"
    target.mkdir()
    (target / "nested.txt").write_text("x", encoding="utf-8")
    helpers.remove_directory(target)
    assert not target.exists()

    vectors_dir = tmp_path / "vectors"
    models_dir = tmp_path / "models"
    vectors_dir.mkdir()
    models_dir.mkdir()
    monkeypatch.setattr(helpers, "vectors_dir", str(vectors_dir))
    monkeypatch.setattr(helpers, "models_dir", str(models_dir))

    helpers.remove_vectors()
    helpers.remove_models()
    assert not vectors_dir.exists()
    assert not models_dir.exists()

    pixel = Image.fromarray(np.asarray([[[255, 0, 0]]], dtype=np.uint8), mode="RGB")
    batch = helpers.export_image_batch([pixel])
    assert batch.shape == (1, 1, 1, 3)
    assert np.allclose(batch.numpy()[0, 0, 0], np.asarray([1.0, 0.0, 0.0], dtype=np.float32))
