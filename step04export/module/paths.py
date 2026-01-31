"""Config loader for ComfyUI node runtime."""

import json
import os
from pathlib import Path

root = Path(__file__).parent.absolute()
config_file_path = os.path.join(root, "config", "config.json")
with open(config_file_path, "r") as f:
    base_config = json.load(f)

prepare_config_path = os.path.join(root, base_config["prepare_config"])
with open(prepare_config_path, "r") as f:
    prepare_config = json.load(f)

maps_dir = os.path.join(root, "maps")
models_dir = os.path.join(root, "models")

training_model = os.path.join(models_dir, base_config["training_model"])
filtered_data = os.path.join(models_dir, base_config["filtered_data"])
interaction_data = os.path.join(models_dir, base_config["interaction_data"])
