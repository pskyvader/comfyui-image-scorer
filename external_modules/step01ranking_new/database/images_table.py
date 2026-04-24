"""Images table operations - index of ranked images (filename-only storage)."""

from typing import Optional, List, Dict, Any
from .schema import get_db_connection
from collections import OrderedDict
from shared.config import config


# Simple LRU cache for image metadata to reduce DB hits.
# Size is configured in ranking subconfig (`ranking_config.json`) as `lru_size`.
try:
    _LRU_SIZE = int(config["ranking"]["lru_size"])
except Exception:
    _LRU_SIZE = 100

_lru_cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()


def _cache_get(key: str) -> Optional[Dict[str, Any]]:
    v = _lru_cache.get(key)
    if v is None:
        return None
    # Move to end to mark recently used
    try:
        _lru_cache.move_to_end(key)
    except Exception:
        pass
    return dict(v)  # return a shallow copy


def _cache_put(key: str, value: Dict[str, Any]) -> None:
    try:
        _lru_cache[key] = dict(value)
        _lru_cache.move_to_end(key)
        # Evict oldest entries
        while len(_lru_cache) > _LRU_SIZE:
            _lru_cache.popitem(last=False)
    except Exception:
        pass


def _cache_invalidate(key: str) -> None:
    try:
        if key in _lru_cache:
            del _lru_cache[key]
    except Exception:
        pass


def add_image(
    filename: str,
    score: float = 0.5,
    confidence: float = 0.3,
    comparison_count: int = 0,
) -> bool:
    """
    Add or update image in database.

    Args:
        filename: Image filename (relative or basename, no path)
        score: Score 0-1 float
        confidence: Confidence 0-1 float
        comparison_count: Number of comparisons

    Returns:
        True if successful
    """
    try:
        with get_db_connection() as conn:
            conn.execute(
                """
                INSERT INTO images(filename, score, confidence, comparison_count)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(filename) DO UPDATE SET 
                    score=excluded.score,
                    confidence=excluded.confidence,
                    comparison_count=excluded.comparison_count
                """,
                (filename, score, confidence, comparison_count),
            )
            conn.commit()
        # Invalidate/refresh cache for this filename
        try:
            _cache_invalidate(filename)
        except Exception:
            pass
        return True
    except Exception as e:
        print(f"Error adding image {filename}: {e}")
        return False


def get_image(filename: str) -> Optional[Dict[str, Any]]:
    """
    Get image metadata by filename.

    Returns:
        Dict with filename, score, confidence, comparison_count, last_compared_at, ranking_generation
        or None if not found
    """
    # Check LRU cache first
    try:
        cached = _cache_get(filename)
        if cached is not None:
            return cached
    except Exception:
        pass

    try:
        with get_db_connection() as conn:
            row = conn.execute(
                "SELECT * FROM images WHERE filename=?",
                (filename,),
            ).fetchone()
            if row:
                result = dict(row)
                try:
                    _cache_put(filename, result)
                except Exception:
                    pass
                return result
    except Exception as e:
        print(f"Error getting image {filename}: {e}")
    return None


def update_image_score(filename: str, score: float) -> bool:
    """Update image score and touch last_compared_at."""
    try:
        with get_db_connection() as conn:
            conn.execute(
                """
                UPDATE images 
                SET score=?, last_compared_at=CURRENT_TIMESTAMP
                WHERE filename=?
                """,
                (score, filename),
            )
            conn.commit()
        # Invalidate cache on update
        try:
            _cache_invalidate(filename)
        except Exception:
            pass
        return True
    except Exception as e:
        print(f"Error updating score for {filename}: {e}")
        return False


def update_image_confidence(
    filename: str, confidence: float, comparison_count: int = None
) -> bool:
    """Update image confidence and optionally comparison count."""
    try:
        with get_db_connection() as conn:
            if comparison_count is not None:
                conn.execute(
                    """
                    UPDATE images 
                    SET confidence=?, comparison_count=?
                    WHERE filename=?
                    """,
                    (confidence, comparison_count, filename),
                )
            else:
                conn.execute(
                    """
                    UPDATE images 
                    SET confidence=?
                    WHERE filename=?
                    """,
                    (confidence, filename),
                )
            conn.commit()
        # Invalidate cache on update
        try:
            _cache_invalidate(filename)
        except Exception:
            pass
        return True
    except Exception as e:
        print(f"Error updating confidence for {filename}: {e}")
        return False


def update_image_score_confidence(
    filename: str, score: float, confidence: float, comparison_count: int = 0
) -> bool:
    """Update image score, confidence, and comparison count together."""
    try:
        with get_db_connection() as conn:
            conn.execute(
                """
                UPDATE images 
                SET score=?, confidence=?, comparison_count=?, last_compared_at=CURRENT_TIMESTAMP
                WHERE filename=?
                """,
                (score, confidence, comparison_count, filename),
            )
            conn.commit()
        # Invalidate cache on update
        try:
            _cache_invalidate(filename)
        except Exception:
            pass
        return True
    except Exception as e:
        print(f"Error updating score/confidence for {filename}: {e}")
        return False


def get_all_images() -> List[Dict[str, Any]]:
    """Get all images from database."""
    try:
        with get_db_connection() as conn:
            rows = conn.execute("SELECT * FROM images").fetchall()
            return [dict(row) for row in rows]
    except Exception as e:
        print(f"Error getting all images: {e}")
        return []


def get_images_by_tier(tier: int) -> List[Dict[str, Any]]:
    """
    Get all images in a specific tier.
    Tier is calculated from score: tier = int(score * 10)
    """
    tier_min = tier / 10.0
    tier_max = (tier + 1) / 10.0

    try:
        with get_db_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM images WHERE score >= ? AND score < ? ORDER BY score",
                (tier_min, tier_max),
            ).fetchall()
            return [dict(row) for row in rows]
    except Exception as e:
        print(f"Error getting images for tier {tier}: {e}")
        return []


def get_image_count() -> int:
    """Get total count of images in database."""
    try:
        with get_db_connection() as conn:
            row = conn.execute("SELECT COUNT(*) as cnt FROM images").fetchone()
            return row["cnt"] if row else 0
    except Exception as e:
        print(f"Error getting image count: {e}")
        return 0


def get_scored_images(
    limit: int = 100, offset: int = 0
) -> tuple[List[Dict[str, Any]], int]:
    """
    Get paginated scored images.

    Returns:
        Tuple of (images_list, total_count)
    """
    try:
        with get_db_connection() as conn:
            total_row = conn.execute(
                "SELECT COUNT(*) as cnt FROM images WHERE score IS NOT NULL"
            ).fetchone()
            total = total_row["cnt"] if total_row else 0

            rows = conn.execute(
                "SELECT * FROM images WHERE score IS NOT NULL ORDER BY score DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
            return [dict(row) for row in rows], total
    except Exception as e:
        print(f"Error getting scored images: {e}")
        return [], 0


def delete_image(filename: str) -> bool:
    """Delete an image from the database by filename.

    Returns:
        True if a row was deleted, False otherwise.
    """
    try:
        with get_db_connection() as conn:
            cur = conn.execute("DELETE FROM images WHERE filename=?", (filename,))
            conn.commit()
        _cache_invalidate(filename)
        return cur.rowcount > 0
    except Exception as e:
        print(f"Error deleting image {filename}: {e}")
        return False

