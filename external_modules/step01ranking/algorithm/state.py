"""Centralised mutable state for the ranking algorithm.

All module-level caches, LRU trackers, and pair metadata live here so that
every sub-module operates on the same shared state without circular imports.
"""

from typing import Any
from collections import deque
import time

from database.images_table import get_all_images
from .constants import IMAGES_CACHE_TTL
from shared.config import config

# ---------------------------------------------------------------------------
# Images cache
# ---------------------------------------------------------------------------
_images_cache: dict[str, Any] = {"data": None, "timestamp": 0.0}


def get_cached_all_images() -> list[dict[str, Any]]:
    """Return cached all_images list, refreshing if stale."""
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


def invalidate_images_cache() -> None:
    """Force the next ``get_cached_all_images`` call to hit the database."""
    global _images_cache
    _images_cache = {"data": None, "timestamp": 0.0}


# ---------------------------------------------------------------------------
# Last pair metadata
# ---------------------------------------------------------------------------
_last_pair_metadata: dict[str, Any] = {}


def get_last_pair_metadata() -> dict[str, Any]:
    """Return metadata from the most recent pair selection."""
    return _last_pair_metadata.copy()


def set_last_pair_metadata(meta: dict[str, Any]) -> None:
    """Replace the stored pair metadata."""
    global _last_pair_metadata
    _last_pair_metadata = meta


# ---------------------------------------------------------------------------
# Loop 2 refinement caches  (chain pair progress + LRU)
# ---------------------------------------------------------------------------

lru_size = config["ranking"]["lru_size"]


chain_pair_progress: dict[tuple[tuple[str, ...], tuple[str, ...]], int] = {}
node_lru_cache: deque[str] = deque(maxlen=lru_size)
chain_lru_cache: deque[int] = deque(maxlen=lru_size)


def clear_old_cache() -> None:
    if len(node_lru_cache) >= lru_size or len(chain_lru_cache) >= lru_size:
        num_to_remove = int(lru_size * 0.75)
        for _ in range(num_to_remove):
            node_lru_cache.popleft()
            chain_lru_cache.popleft()


def update_lru(cache: deque, item: Any, max_size: int = 0) -> None:
    """Track recently-used items; oldest entries are evicted past *max_size*."""
    # Move to end if exists (by removing and re-appending)
    if item in cache:
        cache.remove(item)
    cache.append(item)

    if max_size > 0 and len(cache) >= max_size:
        # todo: change lru to deque, and check if cache length is over max size-1, if so, remove the first 3/4 elements from cache
        # If cache is full (or over max_size-1), remove the first 3/4 elements
        num_to_remove = int(len(cache) * 0.75)
        for _ in range(num_to_remove):
            if cache:
                cache.popleft()
