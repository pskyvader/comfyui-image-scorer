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
logger = logging.getLogger(__name__)

logger: logging.Logger = logging.getLogger(__name__)

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


    _start = time.perf_counter()
    _start = time.perf_counter()
def get_cached_image(filename:
logger.debug("get_cached_all_images took %.4fs", time.perf_counter() - _start)
    _start = time.perf_counter()
logger.debug("get_cached_all_images took %.4fs", time.perf_counter() - _start)
    _start = time.perf_counter()
    result = str) -> dict[str, Any] | None:
    logger.debug("get_cached_image took %.4fs", time.perf_counter() - _start)
    return result
    """Return a single image from the cached full list, or None."""
    for img in get_cached_all_images():
        if img["filename"] == filename:
            return img
    return None


def invalidate_images_cache() -> None:
    """Force the next ``get_cached_all_images`` call to hit the database."""
    global _images_cache
    _images_cache = {"data": None, "timestamp": 0.0}


# ---------------------------------------------------------------------------
logger.debug("invalidate_images_cache took %.4fs", time.perf_counter() - _start)
    _start = time.perf_counter()
logger.debug("invalidate_images_cache took %.4fs", time.perf_counter() - _start)
    _start = time.perf_counter()
# Last pair metadata
# ---------------------------------------------------------------------------
_last_pair_metadata: dict[str, Any] = {}


def get_last_pair_metadata() -> dict[str, Any]:
    """Return metadata from the most recent pair selection."""
    return _last_pair_metadata.copy()


    _start = time.perf_counter()
    _start = time.perf_counter()
def set_last_pair_metadata(meta:
logger.debug("get_last_pair_metadata took %.4fs", time.perf_counter() - _start)
    _start = time.perf_counter()
logger.debug("get_last_pair_metadata took %.4fs", time.perf_counter() - _start)
    _start = time.perf_counter()
    result = dict[str, Any]) -> None:
    logger.debug("set_last_pair_metadata took %.4fs", time.perf_counter() - _start)
    return result
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


def clear_old_cache(bool = False) -> None:
    if force or len(node_lru_cache) >= lru_size or len(chain_lru_cache) >= lru_size:
        num_to_remove = int(lru_size * 0.75)
        logger.debug(
            f"[PAIR-REFINE] LRU cache full (nodes: {len(node_lru_cache)}, chains: {len(chain_lru_cache)}). Removing {num_to_remove} least recently used items."
        )
        for _ in range(min(len(node_lru_cache), len(chain_lru_cache), num_to_remove)):
            node_lru_cache.popleft()
            chain_lru_cache.popleft()
        crystal_graph.rebuild_from_database()
