import os
import shutil
import sys
from pathlib import Path

# Add project root to path to import shared.config
sys.path.insert(0, str(Path(__file__).parent))

from shared.paths import (
    maps_dir,
    comfy_node_path,
    deployment_module_dir,
    training_output_dir,
    prepare_config,
)


def deploy_node():
    if not comfy_node_path:
        raise KeyError("'comfy_node_path' not set in config.json")

    origin = {
        "path": Path(deployment_module_dir),
        "model_path": Path(training_output_dir),
        "maps_path": Path(maps_dir),
        "prepare_config": Path(prepare_config),
    }
    destination = {
        "path": Path(comfy_node_path),
        "models_path": Path(os.path.join(comfy_node_path, "models")),
        "maps_path": Path(os.path.join(comfy_node_path, "maps")),
        "prepare_config": Path(os.path.join(comfy_node_path, "prepare_config.json")),
    }

    # Create destination if not exists
    if destination["path"].exists():
        print("Destination exists, removing old version...")
        shutil.rmtree(destination["path"])

    os.makedirs(destination["path"])
    for key in origin.keys():
        print(f"Copying {key} from {origin[key]} to {destination[key]}...")
        (
            shutil.copytree(origin[key], destination[key])
            if origin[key].is_dir()
            else shutil.copy2(origin[key], destination[key])
        )
    print("Deployment complete.")


if __name__ == "__main__":
    deploy_node()
