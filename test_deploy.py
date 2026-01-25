import json
from pathlib import Path
from shared.config import Config
import deploy


def make_temp_project(tmp_path: Path):
    root = tmp_path / "project"
    root.mkdir()

    # Node source
    src_node = root / "comfyui_custom_nodes" / "ComfyUI-Image-Scorer"
    src_node.mkdir(parents=True)
    for f in ["__init__.py", "nodes.py", "requirements.txt", "README.md"]:
        (src_node / f).write_text("# sample")

    # maps dir
    maps_dir = root / "full_data" / "prepare" / "maps"
    maps_dir.mkdir(parents=True)
    (maps_dir / "a.json").write_text("{}")
    (maps_dir / "b.json").write_text("{}")

    # training output
    training_out = root / "full_data" / "training" / "output"
    training_out.mkdir(parents=True)
    (training_out / "model.onnx").write_text("")

    # config files
    config_dir = root / "config"
    config_dir.mkdir()
    prepare_cfg = config_dir / "prepare_config.json"
    prepare_cfg.write_text(json.dumps({}))
    training_cfg = config_dir / "training_config.json"
    training_cfg.write_text(json.dumps({"output_dir": "full_data/training/output/"}))

    cfg = {
        "root": str(root),
        "comfy_node_path": str(root / "deploy_dest"),
        "maps_dir": "full_data/prepare/maps",
        "prepare_config": "config/prepare_config.json",
        "training_config": "config/training_config.json",
    }
    cfg_path = config_dir / "config.json"
    cfg_path.write_text(json.dumps(cfg))

    return root, cfg_path


def test_deploy_copies_files(tmp_path):
    root, cfg_path = make_temp_project(tmp_path)
    cfg = Config(cfg_path)

    # Monkeypatch deploy module to use our config object
    deploy.config = cfg

    deploy.deploy_node()

    dest = Path(cfg["comfy_node_path"])

    # Check files exist in destination
    assert (dest / "__init__.py").exists()
    assert (dest / "nodes.py").exists()
    # maps copied
    assert (dest / "models" / "maps" / "a.json").exists()


def test_deploy_missing_comfy_node_path_raises(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    config_dir = root / "config"
    config_dir.mkdir()
    cfg = {
        "root": str(root),
        # missing comfy_node_path
        "maps_dir": "full_data/prepare/maps",
        "prepare_config": "config/prepare_config.json",
        "training_config": "config/training_config.json",
    }
    cfg_path = config_dir / "config.json"
    cfg_path.write_text(json.dumps(cfg))

    cfg_obj = Config(cfg_path)
    deploy.config = cfg_obj

    import pytest

    with pytest.raises(KeyError):
        deploy.deploy_node()
