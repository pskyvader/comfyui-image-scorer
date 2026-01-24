import io
import json
import pytest
from unittest.mock import MagicMock, patch
from text_data.text_processing import load_text_index, process_text_files


def test_load_text_index_missing(tmp_path):
    idx = load_text_index(str(tmp_path / "missing.json"))
    assert idx == set()


def test_process_text_files_extracts_entries(tmp_path, monkeypatch):
    image_root = tmp_path / "images"
    image_root.mkdir()
    img_path = image_root / "img1.png"
    img_path.write_bytes(b"fake")
    meta_path = image_root / "img1.json"
    meta_path.write_text(json.dumps({"123": {
        "score": 4.0, 
        "prompt": "hello",
        "width": 512,
        "height": 512,
        "aspect_ratio": 1.0
    }}), encoding="utf-8")

    from shared.config import config
    
    # Patch save to prevent writing to disk
    monkeypatch.setattr("shared.config._save_raw_config", lambda *args: None)
    
    try:
        old_root = config["image_root"]
    except KeyError:
        old_root = None

    try:
        old_prepare = config["prepare"]
    except KeyError:
        old_prepare = None

    if hasattr(old_prepare, "_data"):
        old_prepare = old_prepare._data
    elif old_prepare is not None:
        old_prepare = dict(old_prepare)

    try:
        config["image_root"] = str(image_root)
        # Ensure normalization config exists for extract_text_components
        if "prepare" not in config:
            config["prepare"] = {}
        config["prepare"]["normalization"] = {"cfg_max": 10.0, "steps_max": 100.0}

        processed_files = set()
        buf = io.StringIO()
        error_log: list[dict[str, str]] = []

        with patch("PIL.Image.open") as mock_open:
            mock_img = MagicMock()
            mock_img.__enter__.return_value = mock_img
            mock_img.size = (512, 512)
            mock_open.return_value = mock_img
            
            new_count = process_text_files([(str(img_path), str(meta_path))], processed_files, buf, error_log)

        assert new_count == 1
        assert processed_files == {"img1.png"}
        lines = [ln for ln in buf.getvalue().splitlines() if ln.strip()]
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["file_id"] == "img1.png"
        assert record["score"] == 4.0
        assert record["width"] == 512
        assert error_log == []
    finally:
        if old_root is not None:
            config["image_root"] = old_root
        elif "image_root" in config:
            del config["image_root"]
            
        if old_prepare is not None:
            config["prepare"] = old_prepare
        elif "prepare" in config:
            del config["prepare"]
