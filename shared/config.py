from __future__ import annotations
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Union, Iterator
from collections.abc import MutableMapping

from .io import load_json

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]
ConfigDict = dict[str, Any]
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
CONFIG_FILE: Path = PROJECT_ROOT.joinpath("config", "config.json")
SUB_CONFIG_MAPPING: dict[str, str] = {
    "prepare": "prepare_config",
    "training": "training_config",
    "vector": "vector_config",
    "ranking": "ranking_config",
}


def _get_config_file(path: PathLike) -> Path:
    _start = time.perf_counter()
    p = Path(path)
    if not p.is_absolute():
        p: Path = PROJECT_ROOT.joinpath(p)
    result: Path = p
    return result


def _load_raw_config(path: PathLike) -> ConfigDict:
    _start = time.perf_counter()
    config_file: Path = _get_config_file(path)
    result: ConfigDict
    if not config_file.exists():
        result = {}
    else:
        data, err = load_json(str(config_file), expect=dict, default=None)
        if err:
            result = {}
        else:
            result = data or {}

    return result


def _save_raw_config(data: ConfigDict, path: PathLike) -> None:
    _start = time.perf_counter()
    config_file: Path = _get_config_file(path)
    ensure_dir(config_file.parent)
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def ensure_dir(path: PathLike) -> None:
    _start = time.perf_counter()
    os.makedirs(Path(path), exist_ok=True)


class AutoSaveDict(MutableMapping):
    def __init__(self, data: dict[str, Any], save_callback: Any) -> None:
        _start = time.perf_counter()
        self._data: dict[str, Any] = data
        self._save_callback = save_callback

    def get(self, key: str) -> Any:
        _start = time.perf_counter()
        result: Any = self[key]
        return result

    def __getitem__(self, key: str) -> Any:
        _start = time.perf_counter()
        value = self._data[key]
        result: Any
        if isinstance(value, dict):
            result = AutoSaveDict(value, self._save_callback)
        else:
            result = value
        return result

    def __setitem__(self, key: str, value: Any) -> None:
        _start = time.perf_counter()
        if isinstance(value, AutoSaveDict):
            value = value._data
        self._data[key] = value
        self._save_callback()

    def __delitem__(self, key: str) -> None:
        _start = time.perf_counter()
        del self._data[key]
        self._save_callback()

    def __iter__(self) -> Iterator[str]:
        _start = time.perf_counter()
        result: Iterator[str] = iter(self._data)
        return result

    def __len__(self) -> int:
        _start = time.perf_counter()
        result: int = len(self._data)
        return result

    def copy(self) -> dict[str, Any]:
        _start = time.perf_counter()
        result: dict[str, Any] = self._data.copy()
        return result

    def __repr__(self) -> str:
        _start = time.perf_counter()
        result: str = repr(self._data)
        return result


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

    def __init__(self, config_file: PathLike) -> None:
        _start = time.perf_counter()
        self._root_path: Path = _get_config_file(config_file)
        self._cache: dict[str, Any] = {}
        self._root_data: ConfigDict | None = None
        self._wrappers: dict[str, AutoSaveDict] = {}

    def get(self, key: str) -> Any:
        _start = time.perf_counter()
        result: Any = self[key]

        return result

    def _get_root(self) -> ConfigDict:
        _start = time.perf_counter()
        if self._root_data is None:
            self._root_data = _load_raw_config(self._root_path)
        result: ConfigDict = self._root_data
        return result

    def _save_root(self) -> None:
        _start = time.perf_counter()
        if self._root_data is not None:
            _save_raw_config(self._root_data, self._root_path)

    def _get_sub(self, section: str) -> ConfigDict | None:
        _start = time.perf_counter()
        pointer: str = SUB_CONFIG_MAPPING[section]
        root: dict[str, Any] = self._get_root()
        result: ConfigDict | None
        if pointer not in root:
            result = None
        else:
            path = root[pointer]
            if section not in self._cache:
                self._cache[section] = _load_raw_config(path)
                if section in self._wrappers:
                    del self._wrappers[section]
            result = self._cache[section]

        return result

    def _save_sub(self, section: str) -> None:
        _start = time.perf_counter()
        if section not in self._cache:

            return
        pointer: str = SUB_CONFIG_MAPPING[section]
        root: dict[str, Any] = self._get_root()
        path = root[pointer]
        _save_raw_config(self._cache[section], path)

    def __getitem__(self, key: str) -> Any:
        _start = time.perf_counter()
        result: Any
        if key in SUB_CONFIG_MAPPING:
            data: dict[str, Any] | None = self._get_sub(key)
            if data is None:
                raise KeyError(f"Subconfig '{key}' not configured")
            if key not in self._wrappers:
                self._wrappers[key] = AutoSaveDict(data, lambda: self._save_sub(key))
            result = self._wrappers[key]
        else:
            root: dict[str, Any] = self._get_root()
            if key in root:
                val = root[key]
                result = (
                    AutoSaveDict(val, self._save_root) if isinstance(val, dict) else val
                )
            else:
                found: bool = False
                for section in SUB_CONFIG_MAPPING:
                    data = self._get_sub(section)
                    if data and key in data:
                        sub_wrapper = AutoSaveDict(
                            data, lambda s=section: self._save_sub(s)
                        )
                        result = sub_wrapper[key]
                        found = True
                        break
                if not found:
                    raise KeyError(f"Key '{key}' not found")

        return result

    def __setitem__(self, key: str, value: Any) -> None:
        _start = time.perf_counter()
        if key in SUB_CONFIG_MAPPING:
            if not isinstance(value, dict):
                raise ValueError(
                    f"Subconfig must be a dict, current type: {type(value)}"
                )
            self._cache[key] = value
            if key in self._wrappers:
                del self._wrappers[key]
            self._save_sub(key)
        else:
            root: dict[str, Any] = self._get_root()
            overwritten: bool = False
            for section in SUB_CONFIG_MAPPING:
                data = self._get_sub(section)
                if data is not None and key in data:
                    data[key] = value
                    self._save_sub(section)
                    overwritten = True
                    break
            if not overwritten:
                root[key] = value
                self._save_root()

    def __delitem__(self, key: str) -> None:
        _start = time.perf_counter()
        root: dict[str, Any] = self._get_root()
        if key in root:
            del root[key]
            self._save_root()
        else:
            found: bool = False
            for section in SUB_CONFIG_MAPPING:
                data = self._get_sub(section)
                if data and key in data:
                    del data[key]
                    self._save_sub(section)
                    found = True
                    break
            if not found:
                raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        _start = time.perf_counter()
        keys: set[str] = set(self._get_root().keys())
        keys.update(SUB_CONFIG_MAPPING.keys())
        for section in SUB_CONFIG_MAPPING:
            data = self._get_sub(section)
            if data:
                keys.update(data.keys())
        result: Iterator[str] = iter(keys)

        return result

    def __len__(self) -> int:
        _start = time.perf_counter()
        result: int = len(list(iter(self)))

        return result

    def clear(self) -> None:
        _start = time.perf_counter()
        self._root_data = None
        self._cache.clear()
        self._wrappers.clear()


config = Config(CONFIG_FILE)
