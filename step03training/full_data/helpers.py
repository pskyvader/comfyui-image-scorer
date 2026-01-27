from pathlib import Path
import sys
import importlib
import os
from typing import List, Any, Callable, Optional, Dict
from shared.config import config


def ensure_project_root(
    marker_files: tuple[str, ...] = ("config.json", "setup.py", "README.md")
) -> str:
    p = Path.cwd()
    root: Optional[Path] = None
    for _ in range(8):
        if any((p / m).exists() for m in marker_files):
            root = p
            break
        if p.parent == p:
            root = Path.cwd().parent
            break
        p = p.parent
    if root is None:
        raise FileNotFoundError(
            f"Could not find project root with marker files: {marker_files}"
        )
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root_str


def ensure_dependencies(pkgs: Optional[List[str]] = None) -> List[str]:
    if pkgs is None:
        pkgs = ["numpy", "sklearn", "matplotlib"]
    missing: List[str] = []
    for pkg in pkgs:
        try:
            importlib.import_module(pkg)
        except Exception:
            missing.append(pkg)
    if missing:
        print("Missing packages:", missing)
        print("Install them with: pip install " + " ".join(missing))
    else:
        print("All required packages available")
    return missing


def cast_val(v: Any, caster: Optional[Callable[[Any], Any]]):
    return caster(v) if caster is not None else v


def resolve_path(p: str) -> str:
    if os.path.isabs(p):
        return os.path.abspath(os.path.normpath(p))
    root = ensure_project_root()
    candidate = os.path.abspath(os.path.join(str(root), p))
    return os.path.abspath(os.path.normpath(candidate))


def save_training_config(
    config_data: Dict[str, Any],
) -> Dict[str, Any]:
    config["training"]["recommended_training"] = config_data
    print(f" Training Configuration saved")
    return config_data
