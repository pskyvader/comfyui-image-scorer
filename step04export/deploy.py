import os
import shutil
import sys
import time
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from shared.paths import (
    config_path,
    maps_dir,
    comfy_node_path,
    deployment_module_dir,
    training_output_dir,
)

def verify_deployment(src, dst):
    """Checks if all items in src exist in dst."""
    src_items = set(os.listdir(src))
    dst_items = set(os.listdir(dst))
    missing = src_items - dst_items
    if missing:
        print(f"  [ERROR] Missing items in destination: {missing}")
    else:
        print(f"  [SUCCESS] All {len(src_items)} items verified in {dst.name}")

def deploy_node():
    if not comfy_node_path:
        raise KeyError("'comfy_node_path' not set in config.json")

    # Use .resolve() to ensure absolute paths and prevent Windows relative path locks
    origin = {
        "path": Path(deployment_module_dir).resolve(),
        "models_path": Path(training_output_dir).resolve(),
        "maps_path": Path(maps_dir).resolve(),
        "config": Path(config_path).resolve(),
    }
    
    node_root = Path(comfy_node_path).resolve()
    destination = {
        "path": node_root,
        "models_path": node_root / "models",
        "maps_path": node_root / "maps",
        "config": node_root / "config",
    }

    # 1. Clear destination
    if node_root.exists():
        print(f"Cleaning existing node at {node_root}...")
        shutil.rmtree(node_root, ignore_errors=True)
        # Small delay to let Windows release file handles
        time.sleep(0.2)

    # 2. Create root immediately to 'lock' the path for this script
    node_root.mkdir(parents=True, exist_ok=True)

    # 3. Copy and Verify
    for key, src in origin.items():
        dst = destination[key]
        
        if not src.exists():
            print(f"  [SKIP] Source does not exist: {src}")
            continue

        print(f"Copying {key}...")
        try:
            # dirs_exist_ok=True handles folder merging if 'path' created subdirs
            shutil.copytree(src, dst, dirs_exist_ok=True)
            verify_deployment(src, dst)
        except PermissionError as e:
            print(f"  [CRITICAL] Permission Denied on {key}: {e}")
            print("  [TIP] Close ComfyUI, File Explorer, or VS Code windows looking at the destination.")
        except Exception as e:
            print(f"  [ERROR] Failed to copy {key}: {e}")

    print("\nDeployment sequence finished.")

if __name__ == "__main__":
    deploy_node()
