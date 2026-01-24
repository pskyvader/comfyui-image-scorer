from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Dict, Any, Iterable, Union, Optional, Iterator
from collections.abc import MutableMapping

from shared.io import load_json

PathLike = Union[str, Path]
ConfigDict = Dict[str, Any]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = PROJECT_ROOT.joinpath("config", "config.json")
CONFIG_POINTERS: Iterable[str] = ("prepare_config", "training_config")
SUB_CONFIG_MAPPING = {"prepare": "prepare_config", "training": "training_config"}


def _get_config_file(path: PathLike) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT.joinpath(p)
    return p


def _load_raw_config(path: PathLike) -> ConfigDict:
    config_file = _get_config_file(path)
    if not config_file.exists():
        return {}
    data, err = load_json(str(config_file), expect=dict, default=None)
    if err:
        return {}
    return data or {}


def _save_raw_config(data: ConfigDict, path: PathLike) -> None:
    config_file = _get_config_file(path)
    ensure_dir(config_file.parent)
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def ensure_dir(path: PathLike) -> None:
    os.makedirs(Path(path), exist_ok=True)


_sentinel = object()


class AutoSaveDict(MutableMapping):
    def __init__(self, data: Dict[str, Any], save_callback: callable):
        self._data = data
        self._save_callback = save_callback

    def get(self, key: str, default: Any = _sentinel) -> Any:
        if default is not _sentinel:
            # We strictly ban default values to avoid hidden hardcoded behavior
            raise ValueError(f"Providing a default value for '{key}' is not allowed. All config values must be present in the configuration files.")
        
        # If no default is provided, it behaves like __getitem__ (raises KeyError if missing)
        return self[key]

    def __getitem__(self, key: str) -> Any:
        value = self._data[key]
        if isinstance(value, dict):
            return AutoSaveDict(value, self._save_callback)
        return value

    def __setitem__(self, key: str, value: Any) -> None:
        if isinstance(value, AutoSaveDict):
            value = value._data
        self._data[key] = value
        self._save_callback()

    def __delitem__(self, key: str) -> None:
        del self._data[key]
        self._save_callback()

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def copy(self) -> Dict[str, Any]:
        return self._data.copy()

    def __repr__(self) -> str:
        return repr(self._data)


class Config(MutableMapping):
    """
    Configuration Manager.
    
    STRICT POLICY:
    It is strictly forbidden to provide a default value when accessing configuration keys.
    The goal is to ensure that ALL configuration values are loaded explicitly from the 
    configuration files (JSON) and absolutely nowhere else.
    
    This prevents hidden hardcoded values in the codebase and ensures that the
    configuration files are the single source of truth.
    
    Usage of .get(key, default) will raise a ValueError.
    Use .get(key) (without default) or direct access [key]. Both will raise KeyError if missing.
    """
    def __init__(self, config_file: PathLike = CONFIG_FILE):
        self._root_path = _get_config_file(config_file)
        self._cache: Dict[str, Any] = {}
        self._root_data: Optional[ConfigDict] = None
        self._wrappers: Dict[str, AutoSaveDict] = {}

    def get(self, key: str, default: Any = _sentinel) -> Any:
        if default is not _sentinel:
             # We strictly ban default values to avoid hidden hardcoded behavior
            raise ValueError(f"Providing a default value for '{key}' is not allowed. All config values must be present in the configuration files.")
        
        # If no default is provided, it behaves like __getitem__ (raises KeyError if missing)
        return self[key]

    def _get_root(self) -> ConfigDict:
        if self._root_data is None:
            self._root_data = _load_raw_config(self._root_path)
        return self._root_data

    def _save_root(self) -> None:
        if self._root_data is not None:
            _save_raw_config(self._root_data, self._root_path)

    def _get_sub(self, section: str) -> Optional[ConfigDict]:
        pointer = SUB_CONFIG_MAPPING.get(section)
        if not pointer:
            return None

        root = self._get_root()
        if pointer not in root:
            return None

        path = root[pointer]
        if section not in self._cache:
            self._cache[section] = _load_raw_config(path)
            # If valid wrapper exists but points to old data (unlikely if strictly controlled), invalidating is safe.
            if section in self._wrappers:
                del self._wrappers[section]

        return self._cache[section]

    def _save_sub(self, section: str) -> None:
        if section not in self._cache:
            return

        pointer = SUB_CONFIG_MAPPING.get(section)
        root = self._get_root()
        path = root[pointer]
        _save_raw_config(self._cache[section], path)

    def __getitem__(self, key: str) -> Any:
        # Check subconfig mappings
        if key in SUB_CONFIG_MAPPING:
            data = self._get_sub(key)
            if data is None:
                raise KeyError(f"Subconfig '{key}' not configured")
            if key not in self._wrappers:
                self._wrappers[key] = AutoSaveDict(data, lambda: self._save_sub(key))
            return self._wrappers[key]

        # Check root
        root = self._get_root()
        if key in root:
            val = root[key]
            if isinstance(val, dict):
                # We could cache root wrappers too if needed, but usually root items are simple strings/paths
                return AutoSaveDict(val, self._save_root)
            return val

        # Check deep keys in subconfigs
        for section in SUB_CONFIG_MAPPING:
            data = self._get_sub(section)
            if data and key in data:
                # We return a wrapper for the sub-dict, but this wrapper is ephemeral.
                # Logic for caching ephemeral deep wrappers is complex.
                # Users generally shouldn't rely on 'is' identity for deep random access.
                sub_wrapper = AutoSaveDict(data, lambda s=section: self._save_sub(s))
                return sub_wrapper[key]

        raise KeyError(f"Key '{key}' not found")

    def __setitem__(self, key: str, value: Any) -> None:
        # If key is section name
        if key in SUB_CONFIG_MAPPING:
            if not isinstance(value, dict):
                raise ValueError(f"Subconfig must be a dict, current type: {type(value)}")
            self._cache[key] = value
            if key in self._wrappers:
                del self._wrappers[key]
            self._save_sub(key)
            return

        # If key in root
        root = self._get_root()

        # Check if overwrite subconfig deep key
        for section in SUB_CONFIG_MAPPING:
            data = self._get_sub(section)
            if data is not None and key in data:
                data[key] = value
                self._save_sub(section)
                return

        root[key] = value
        self._save_root()

    def __delitem__(self, key: str) -> None:
        root = self._get_root()
        if key in root:
            del root[key]
            self._save_root()
            return

        for section in SUB_CONFIG_MAPPING:
            data = self._get_sub(section)
            if data and key in data:
                del data[key]
                self._save_sub(section)
                return

        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        keys = set(self._get_root().keys())
        keys.update(SUB_CONFIG_MAPPING.keys())
        for section in SUB_CONFIG_MAPPING:
            data = self._get_sub(section)
            if data:
                keys.update(data.keys())
        return iter(keys)

    def __len__(self) -> int:
        return len(list(iter(self)))

    def clear(self) -> None:
        """Clear cache to force reload from disk."""
        self._root_data = None
        self._cache.clear()
        self._wrappers.clear()


# Global singleton
config = Config(CONFIG_FILE)
