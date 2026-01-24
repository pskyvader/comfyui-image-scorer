import os
import shutil
import sys
import json
from pathlib import Path

# Add project root to path to import shared.config
sys.path.insert(0, str(Path(__file__).parent))

from shared.config import config

def deploy_node():
    root = Path(config["root"])
    dest_path_str = config.get("comfy_node_path")
    
    if not dest_path_str:
        print("Error: 'comfy_node_path' not set in config.json")
        return

    dest_path = Path(dest_path_str)
    
    print(f"Deploying ComfyUI Node to: {dest_path}")
    
    # Create destination if not exists
    os.makedirs(dest_path, exist_ok=True)
    
    # 1. Copy Code Structure
    src_node_root = root / "comfyui_custom_nodes" / "ComfyUI-Image-Scorer"
    
    # Files to copy from root of node
    root_files = ["__init__.py", "nodes.py", "requirements.txt", "README.md"]
    for f in root_files:
        src = src_node_root / f
        dst = dest_path / f
        if src.exists():
            shutil.copy2(src, dst)
            print(f"Copied {f}")
        else:
            print(f"Warning: Source file {f} not found.")

    # Copy lib directory (recurisve)
    src_lib = src_node_root / "lib"
    dst_lib = dest_path / "lib"
    if src_lib.exists():
        if dst_lib.exists():
            shutil.rmtree(dst_lib)
        shutil.copytree(src_lib, dst_lib)
        print("Copied lib/ directory")
        
    # 2. Copy Models (Binaries)
    # Origin: training/output -> Destination: models/bin
    dst_models_bin = dest_path / "models" / "bin"
    os.makedirs(dst_models_bin, exist_ok=True)
    
    training_out = root / "training" / "output"

    # Always copy model and all feature/interaction caches
    model_files = [
        "model.onnx",
        "model.npz",
        "processed_data_cache.npz",
        "filtered_data_cache.npz",
        "interaction_data_cache.npz",
    ]
    for mf in model_files:
        src = training_out / mf
        dst = dst_models_bin / mf
        if src.exists():
            shutil.copy2(src, dst)
            print(f"Copied model/cached file: {mf}")
        else:
            print(f"Warning: Model/cache file {mf} not found in {training_out}")

    # 3. Copy Maps
    # Origin: prepare/maps -> Destination: models/maps
    dst_models_maps = dest_path / "models" / "maps"
    os.makedirs(dst_models_maps, exist_ok=True)
    
    maps_dir = root / config["maps_dir"]
    if maps_dir.exists():
        count = 0
        for map_file in maps_dir.glob("*.json"):
            shutil.copy2(map_file, dst_models_maps / map_file.name)
            count += 1
        print(f"Copied {count} map files to models/maps")
    else:
        print(f"Warning: Maps directory not found at {maps_dir}")

    # Copy an appropriate prepare_config for the node (prefer node-specific override)
    comfy_override = root / "config" / "comfy_prepare_config.json"
    if comfy_override.exists():
        prepare_src = comfy_override
    else:
        prepare_src = root / config["prepare_config"]

    if prepare_src.exists():
        try:
            # Always copy to destination as 'prepare_config.json' so the node can find it by name
            shutil.copy2(prepare_src, dest_path / "prepare_config.json")
            print(f"Copied prepare config: {prepare_src.name} -> prepare_config.json")
        except Exception as e:
            print(f"Warning: Failed to copy prepare config: {e}")
    else:
        print(f"Warning: prepare_config not found at {prepare_src}")

    print("Deployment complete.")

if __name__ == "__main__":
    deploy_node()
