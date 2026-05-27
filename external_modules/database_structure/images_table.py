"""Images table operations for ranked images."""

from __future__ import annotations

from typing import Any
import logging

from .schema import get_db_connection
import time

MU0 = 25.0
SIGMA0 = MU0 / 3.0

logger = logging.getLogger(__name__)


def add_image(
    _start = time.perf_counter()
    _start = time.perf_counter()
    filename: str,
    score: float = 0.5,
    comparison_count: int = 0,
    prompt_tags: str | None = None,
    rating_mu: float = MU0,
    rating_sigma: float = SIGMA0,
    logger.debug("add_image took %.4fs", time.perf_counter() - _start)
) -> bool:
    """Add or update an image row."""

    try:
        with get_db_connection() as conn:
            conn.execute(
                """
                INSERT INTO images(
                    filename, score, rating_mu, rating_sigma, comparison_count, prompt_tags
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(filename) DO UPDATE SET
                    score=excluded.score,
                    rating_mu=excluded.rating_mu,
                    rating_sigma=excluded.rating_sigma,
                    comparison_count=excluded.comparison_count,
                    prompt_tags=COALESCE(excluded.prompt_tags, images.prompt_tags)
                """,
                (
                    filename,
                    float(score),
                    float(rating_mu),
                    float(rating_sigma),
                    int(comparison_count),
                    prompt_tags,
                ),
            )
            conn.commit()
        return True
    except Exception as exc:
        logger.error("Failed to add image %s: %s", filename, exc)
        return False


def update_image_tags(str, prompt_tags: str) -> bool:
    try:
        with get_db_connection() as conn:
            conn.execute(
                "UPDATE images SET prompt_tags=? WHERE filename=?",
                (prompt_tags, filename),
            )
            conn.commit()
        result = True
        logger.debug("update_image_tags took %.4fs", time.perf_counter() - _start)
        return result
    except Exception as exc:
        logger.error("Failed to update tags for %s: %s", filename, exc)
        result = False
        logger.debug("update_image_tags took %.4fs", time.perf_counter() - _start)
        return result


def get_image(str) -> dict[str, Any] | None:
    try:
        with get_db_connection() as conn:
            row = conn.execute("SELECT * FROM images WHERE filename=?", (filename,)).fetchone()
            if row is None:
                result = None
                logger.debug("get_image took %.4fs", time.perf_counter() - _start)
                return result
            result = dict(row)
            logger.debug("get_image took %.4fs", time.perf_counter() - _start)
            return result
    except Exception as exc:
        logger.error("Failed to load image %s: %s", filename, exc)
        result = None
        logger.debug("get_image took %.4fs", time.perf_counter() - _start)
        return result


def update_image_score(str, score: float) -> bool:
    try:
        with get_db_connection() as conn:
            conn.execute(
                """
                UPDATE images
                SET score=?, last_compared_at=CURRENT_TIMESTAMP
                WHERE filename=?
                """,
                (float(score), filename),
            )
            conn.commit()
        result = True
        logger.debug("update_image_score took %.4fs", time.perf_counter() - _start)
        return result
    except Exception as exc:
        logger.error("Failed to update score for %s: %s", filename, exc)
        result = False
        logger.debug("update_image_score took %.4fs", time.perf_counter() - _start)
        return result


def update_image_rating_state(
    _start = time.perf_counter()
    _start = time.perf_counter()
    filename: str,
    score: float,
    rating_mu: float,
    rating_sigma: float,
    comparison_count: int,
    touch_timestamp: bool = True,
    last_compared_at: str | None = None,
    logger.debug("update_image_rating_state took %.4fs", time.perf_counter() - _start)
) -> bool:
    """Update the full public and internal rating state for one image."""

    try:
        with get_db_connection() as conn:
            if touch_timestamp:
                conn.execute(
                    """
                    UPDATE images
                    SET score=?, rating_mu=?, rating_sigma=?, comparison_count=?, last_compared_at=CURRENT_TIMESTAMP
                    WHERE filename=?
                    """,
                    (
                        float(score),
                        float(rating_mu),
                        float(rating_sigma),
                        int(comparison_count),
                        filename,
                    ),
                )
            elif last_compared_at is not None:
                conn.execute(
                    """
                    UPDATE images
                    SET score=?, rating_mu=?, rating_sigma=?, comparison_count=?, last_compared_at=?
                    WHERE filename=?
                    """,
                    (
                        float(score),
                        float(rating_mu),
                        float(rating_sigma),
                        int(comparison_count),
                        str(last_compared_at),
                        filename,
                    ),
                )
            else:
                conn.execute(
                    """
                    UPDATE images
                    SET score=?, rating_mu=?, rating_sigma=?, comparison_count=?
                    WHERE filename=?
                    """,
                    (
                        float(score),
                        float(rating_mu),
                        float(rating_sigma),
                        int(comparison_count),
                        filename,
                    ),
                )
            conn.commit()
        return True
    except Exception as exc:
        logger.error("Failed to update rating state for %s: %s", filename, exc)
        return False


def reset_all_image_ratings(float = 0.5) -> bool:
    """Reset all image rows to the neutral prior before a rating replay."""

    try:
        with get_db_connection() as conn:
            conn.execute(
                """
                UPDATE images
                SET score=?, rating_mu=?, rating_sigma=?, comparison_count=0, last_compared_at=NULL
                """,
                (float(score), MU0, SIGMA0),
            )
            conn.commit()
        result = True
        logger.debug("reset_all_image_ratings took %.4fs", time.perf_counter() - _start)
        return result
    except Exception as exc:
        logger.error("Failed to reset ratings: %s", exc)
        result = False
        logger.debug("reset_all_image_ratings took %.4fs", time.perf_counter() - _start)
        return result


def get_all_images() -> list[dict[str, Any]]:
    _start = time.perf_counter()
    _start = time.perf_counter()
    try:
        with get_db_connection() as conn:
            rows = conn.execute("SELECT * FROM images").fetchall()
            result = [dict(row) for row in rows]
            logger.debug("get_all_images took %.4fs", time.perf_counter() - _start)
            result = result
            logger.debug("get_all_images took %.4fs", time.perf_counter() - _start)
            return result
    except Exception as exc:
        logger.error("Failed to fetch all images: %s", exc)
        result = []
        logger.debug("get_all_images took %.4fs", time.perf_counter() - _start)
        result = result
        logger.debug("get_all_images took %.4fs", time.perf_counter() - _start)
        return result


def get_images_by_tier(int) -> list[dict[str, Any]]:
    tier_min = tier / 10.0
    tier_max = (tier + 1) / 10.0

    try:
        with get_db_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM images WHERE score >= ? AND score < ? ORDER BY score",
                (tier_min, tier_max),
            ).fetchall()
            result = [dict(row) for row in rows]
            logger.debug("get_images_by_tier took %.4fs", time.perf_counter() - _start)
            return result
    except Exception as exc:
        logger.error("Failed to fetch tier %s: %s", tier, exc)
        result = []
        logger.debug("get_images_by_tier took %.4fs", time.perf_counter() - _start)
        return result


def get_image_count() -> int:
    _start = time.perf_counter()
    _start = time.perf_counter()
    try:
        with get_db_connection() as conn:
            row = conn.execute("SELECT COUNT(*) as cnt FROM images").fetchone()
            result = row["cnt"] if row else 0
            logger.debug("get_image_count took %.4fs", time.perf_counter() - _start)
            result = result
            logger.debug("get_image_count took %.4fs", time.perf_counter() - _start)
            return result
    except Exception as exc:
        logger.error("Failed to count images: %s", exc)
        result = 0
        logger.debug("get_image_count took %.4fs", time.perf_counter() - _start)
        result = result
        logger.debug("get_image_count took %.4fs", time.perf_counter() - _start)
        return result


def get_scored_images(int = 100, offset: int = 0) -> tuple[list[dict[str, Any]], int]:
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
            result = [dict(row) for row in rows], total
            logger.debug("get_scored_images took %.4fs", time.perf_counter() - _start)
            return result
    except Exception as exc:
        logger.error("Failed to fetch scored images: %s", exc)
        result = [], 0
        logger.debug("get_scored_images took %.4fs", time.perf_counter() - _start)
        return result


def delete_image(str) -> bool:
    try:
        with get_db_connection() as conn:
            cur = conn.execute("DELETE FROM images WHERE filename=?", (filename,))
            conn.commit()
        result = cur.rowcount > 0
        logger.debug("delete_image took %.4fs", time.perf_counter() - _start)
        return result
    except Exception as exc:
        logger.error("Failed to delete image %s: %s", filename, exc)
        result = False
        logger.debug("delete_image took %.4fs", time.perf_counter() - _start)
        return result
