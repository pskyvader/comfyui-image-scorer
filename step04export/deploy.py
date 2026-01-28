import os
import shutil
import sys
from pathlib import Path

# Add project root to path to import shared.config
# Go up 2 levels from step04export/deploy.py to get to project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

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

    # Remove destination if it exists
    if destination["path"].exists():
        print("Destination exists, removing old version...")
        shutil.rmtree(destination["path"])

    # Copy module directory
    print(f"Copying module from {origin['path']} to {destination['path']}...")
    shutil.copytree(origin["path"], destination["path"])
    
    # Create subdirectories and copy supporting files
    for key in ["models_path", "maps_path"]:
        if key in destination:
            os.makedirs(destination[key], exist_ok=True)
    
    # Copy model files if they exist
    if origin["model_path"].exists():
        print(f"Copying models from {origin['model_path']} to {destination['models_path']}...")
        for file_path in origin["model_path"].glob("*"):
            if file_path.is_file():
                shutil.copy2(file_path, destination["models_path"])
            elif file_path.is_dir() and file_path.name not in ["output", "temp"]:
                shutil.copytree(file_path, destination["models_path"] / file_path.name, dirs_exist_ok=True)
    
    # Copy maps directory if it exists
    if origin["maps_path"].exists():
        print(f"Copying maps from {origin['maps_path']} to {destination['maps_path']}...")
        shutil.copytree(origin["maps_path"], destination["maps_path"], dirs_exist_ok=True)
    
    # Copy prepare config if it exists
    if origin["prepare_config"].exists():
        print(f"Copying prepare_config from {origin['prepare_config']} to {destination['prepare_config']}...")
        shutil.copy2(origin["prepare_config"], destination["prepare_config"])
    
    print("Deployment complete.")


if __name__ == "__main__":
    deploy_node()
