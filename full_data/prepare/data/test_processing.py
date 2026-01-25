import os
from pathlib import Path
from shared.config import Config
import json
import shutil
import tempfile
import pytest

import full_data.prepare.data.processing as processing


def make_temp_project(tmp_path: Path):
    # Create a minimal project structure with files and dirs to be removed
    root = tmp_path / "project"
    root.mkdir()

    # Create directories
    (root / "full_data" / "prepare" / "output").mkdir(parents=True)
    (root / "full_data" / "prepare" / "maps").mkdir(parents=True)
    (root / "text_data" / "prepare" / "output").mkdir(parents=True)

    # Create files
    vectors = root / "full_data" / "prepare" / "output" / "vectors.jsonl"
    scores = root / "full_data" / "prepare" / "output" / "scores.jsonl"
    index = root / "full_data" / "prepare" / "output" / "index.json"
    text_data = root / "text_data" / "prepare" / "output" / "text_data.jsonl"
    text_index = root / "text_data" / "prepare" / "output" / "text_index.json"

    for f in [vectors, scores, index, text_data, text_index]:
        f.write_text("[]")

    # Add a sample map file
    (root / "full_data" / "prepare" / "maps" / "sample_map.json").write_text("{}")

    # Create minimal config files
    config_dir = root / "config"
    config_dir.mkdir()
    prepare_cfg = config_dir / "prepare_config.json"
    prepare_cfg.write_text(json.dumps({"vector_schema": {"slots": {}}}))
    training_cfg = config_dir / "training_config.json"
    training_cfg.write_text(json.dumps({"output_dir": "full_data/training/output/"}))

    # Create global config.json
    cfg = {
        "root": str(root),
        "maps_dir": "full_data/prepare/maps",
        "prepare_config": "config/prepare_config.json",
        "training_config": "config/training_config.json",
        "vectors_file": "full_data/prepare/output/vectors.jsonl",
        "scores_file": "full_data/prepare/output/scores.jsonl",
        "index_file": "full_data/prepare/output/index.json",
        "text_data_file": "text_data/prepare/output/text_data.jsonl",
        "text_index_file": "text_data/prepare/output/text_index.json",
    }
    cfg_path = config_dir / "config.json"
    cfg_path.write_text(json.dumps(cfg))

    return root, cfg_path


def test_remove_existing_outputs_removes_files_and_dirs(tmp_path):
    root, cfg_path = make_temp_project(tmp_path)

    # Create a fresh Config that points to our tmp config.json
    cfg = Config(cfg_path)

    # Monkeypatch the module to use our config instance
    processing.config = cfg

    # Ensure files exist before
    assert (root / "full_data" / "prepare" / "output" / "vectors.jsonl").exists()
    assert (root / "full_data" / "prepare" / "maps").exists()

    processing.remove_existing_outputs()

    # Files and directory should be removed
    assert not (root / "full_data" / "prepare" / "output" / "vectors.jsonl").exists()
    assert not (root / "full_data" / "prepare" / "maps").exists()


def test_remove_existing_outputs_handles_nonexistent_paths(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    config_dir = root / "config"
    config_dir.mkdir()
    cfg = {
        "root": str(root),
        "maps_dir": "full_data/prepare/maps",
        "prepare_config": "config/prepare_config.json",
        "training_config": "config/training_config.json",
        "vectors_file": "full_data/prepare/output/vectors.jsonl",
        "scores_file": "full_data/prepare/output/scores.jsonl",
        "index_file": "full_data/prepare/output/index.json",
        "text_data_file": "text_data/prepare/output/text_data.jsonl",
        "text_index_file": "text_data/prepare/output/text_index.json",
    }
    cfg_path = config_dir / "config.json"
    cfg_path.write_text(json.dumps(cfg))

    cfg_obj = Config(cfg_path)
    processing.config = cfg_obj

    # Nothing should raise even if paths do not exist
    processing.remove_existing_outputs()
