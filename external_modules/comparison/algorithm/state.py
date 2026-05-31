"""Centralised mutable state for the ranking algorithm.

All module-level caches, LRU trackers, and pair metadata live here so that
every sub-module operates on the same shared state without circular imports.
"""

from typing import Any
from collections import deque
import time
import logging

from external_modules.database_structure.images_table import get_all_images
from .constants import IMAGES_CACHE_TTL
from shared.config import config
from shared.graph.crystal_graph import crystal_graph

logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Images cache
# ---------------------------------------------------------------------------
_images_cache: dict[str, Any] = {"data": None, "timestamp": 0.0}


def get_cached_all_images() -> list[dict[str, Any]]:
    """Return cached all_images list, refreshing if stale."""
    _start = time.perf_counter()
    global _images_cache
    now = time.time()
    if (
        _images_cache["data"] is not None
        and (now - _images_cache["timestamp"]) < IMAGES_CACHE_TTL
    ):
        return _images_cache["data"]

    data = get_all_images()
    _images_cache = {"data": data, "timestamp": now}
    return data


def get_cached_image(filename: str) -> dict[str, Any] | None:
    """Return a single image from the cached full list, or None."""
    _start = time.perf_counter()
    result: dict[str, Any] | None = None
    for img in get_cached_all_images():
        if img["filename"] == filename:
            result = img
            break
    return result


def invalidate_images_cache() -> None:
    """Force the next get_cached_all_images call to hit the database."""
    _start = time.perf_counter()
    global _images_cache
    _images_cache = {"data": None, "timestamp": 0.0}


# ---------------------------------------------------------------------------
# Last pair metadata
# ---------------------------------------------------------------------------
_last_pair_metadata: dict[str, Any] = {}


def get_last_pair_metadata() -> dict[str, Any]:
    """Return metadata from the most recent pair selection."""
    global _last_pair_metadata
    result: dict[str, Any] = _last_pair_metadata.copy()
    return result


def set_last_pair_metadata(meta: dict[str, Any]) -> None:
    """Replace the stored pair metadata."""
    global _last_pair_metadata
    _last_pair_metadata = meta
