"""Comparisons table operations - full history of all comparisons."""

import logging
from typing import Any
from datetime import datetime, timedelta, timezone
from .schema import get_db_connection

logger = logging.getLogger(__name__)


def _canonicalize_pair(filename_a: str, filename_b: str) -> tuple[str, str]:
    """Store comparison pairs in a stable lexical order."""
    first = str(filename_a)
    second = str(filename_b)
    sorted_pair = sorted((first, second))
    return (sorted_pair[0], sorted_pair[1])


def add_historical_comparison(
    filename_a: str,
    filename_b: str,
    winner: str,
    timestamp: str,
    weight: float = 1.0,
    transitive_depth: int = 0,
) -> int:
    """Insert a historical comparison if it doesn't already exist.

    Returns:
        The insert ID if successful or already exists, 0 otherwise.
    """
    filename_a = str(filename_a)
    filename_b = str(filename_b)
    winner = str(winner)
    if winner not in (filename_a, filename_b):
        return 0
    filename_a, filename_b = _canonicalize_pair(filename_a, filename_b)
    timestamp = str(timestamp)

    try:
        with get_db_connection() as conn:
            # Check if exists
            exists = conn.execute(
                """
                SELECT id FROM comparisons 
                WHERE timestamp = ? AND 
                ((filename_a = ? AND filename_b = ?) OR (filename_a = ? AND filename_b = ?))
                AND winner = ?
                AND weight = ?
                AND transitive_depth = ?
                """,
                (
                    timestamp,
                    filename_a,
                    filename_b,
                    filename_b,
                    filename_a,
                    winner,
                    weight,
                    transitive_depth,
                ),
            ).fetchone()

            if exists:
                return exists["id"]

            cur = conn.execute(
                """
                INSERT INTO comparisons(filename_a, filename_b, winner, timestamp, weight, transitive_depth)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (filename_a, filename_b, winner, timestamp, weight, transitive_depth),
            )
            conn.commit()
            return cur.lastrowid or 0
    except Exception as e:
        logger.debug(f"Failed to add historical comparison during rebuild: {e}")
        return 0


def add_comparison(
    filename_a: str,
    filename_b: str,
    winner: str,
    weight: float = 1.0,
    transitive_depth: int = 0,
    timestamp: str | None = None,
) -> int:
    """
    Record a comparison result.

    Args:
        filename_a: First image filename
        filename_b: Second image filename
        winner: Winning image filename (must be one of the above)
        weight: Comparison weight (1.0 for direct, <1.0 for transitive)
        transitive_depth: Depth of transitive inference (0 for direct comparison)
        timestamp: Optional ISO timestamp. When omitted, a new UTC timestamp is generated.

    Returns:
        The insert ID if successful, 0 otherwise
    """
    filename_a = str(filename_a)
    filename_b = str(filename_b)
    winner = str(winner)
    if winner not in (filename_a, filename_b):
        logger.error(f"Error: winner must be one of the compared images")
        return 0
    filename_a, filename_b = _canonicalize_pair(filename_a, filename_b)
    timestamp_value = str(timestamp) if timestamp else datetime.now(timezone.utc).isoformat()

    try:
        with get_db_connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO comparisons(filename_a, filename_b, winner, timestamp, weight, transitive_depth)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    filename_a,
                    filename_b,
                    winner,
                    timestamp_value,
                    weight,
                    transitive_depth,
                ),
            )
            conn.commit()
            return cur.lastrowid or 0
    except Exception as e:
        logger.error(f"Error adding comparison: {e}")
        return 0


def delete_duplicate_comparisons() -> int:
    """Remove exact duplicate comparison rows while keeping the earliest copy."""
    try:
        with get_db_connection() as conn:
            cur = conn.execute(
                """
                DELETE FROM comparisons
                WHERE id NOT IN (
                    SELECT MIN(id)
                    FROM comparisons
                    GROUP BY
                        CASE
                            WHEN filename_a < filename_b THEN filename_a
                            ELSE filename_b
                        END,
                        CASE
                            WHEN filename_a < filename_b THEN filename_b
                            ELSE filename_a
                        END,
                        winner,
                        timestamp,
                        weight,
                        transitive_depth
                )
                """
            )
            conn.commit()
            return max(cur.rowcount, 0)
    except Exception as e:
        logger.debug(f"Failed to delete duplicate comparisons: {e}")
        return 0


def get_recent_comparisons(
    filename: str, days: int = 30, limit: int = 100
) -> list[dict[str, Any]]:
    """
    Get recent comparisons for an image (within N days).

    Used for confidence calculation (stability over recent window).
    """
    try:
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        with get_db_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM comparisons 
                WHERE (filename_a=? OR filename_b=?)
                AND timestamp > ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (filename, filename, cutoff_date.isoformat(), limit),
            ).fetchall()
            return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Error getting recent comparisons for {filename}: {e}")
        return []


def get_comparison_count(filename: str) -> int:
    """Count total comparisons for an image."""
    try:
        with get_db_connection() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) as cnt FROM comparisons 
                WHERE filename_a=? OR filename_b=?
                """,
                (filename, filename),
            ).fetchone()
            return row["cnt"] if row else 0
    except Exception as e:
        logger.error(f"Error getting comparison count for {filename}: {e}")
        return 0


def get_total_comparisons() -> int:
    """Get total count of all comparisons in database."""
    try:
        with get_db_connection() as conn:
            row = conn.execute("SELECT COUNT(*) as cnt FROM comparisons").fetchone()
            return row["cnt"] if row else 0
    except Exception as e:
        logger.error(f"Error getting total comparisons: {e}")
        return 0


def get_skipped_comparison_count() -> int:
    """Count all comparisons recorded with a weight less than 1.0."""
    try:
        with get_db_connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM comparisons WHERE weight < 1.0"
            ).fetchone()
            return row["cnt"] if row else 0
    except Exception as e:
        logger.error(f"Error getting skipped comparison count: {e}")
        return 0


def get_all_comparisons(weight: float | None = None) -> list[dict[str, Any]]:
    """Get all comparisons from the database."""
    query: str = "SELECT * FROM comparisons"
    query += " WHERE TRUE"
    variables: list[Any] = []
    if weight is not None:
        query += " AND weight=?"
        variables.append(weight)

    # cast parameters to tuple for execute
    parameters: tuple[Any, ...] = tuple(variables)

    try:
        with get_db_connection() as conn:
            rows = conn.execute(query, parameters).fetchall()
            return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Error getting all comparisons: {e}")
        return []


def deduplicate_comparisons() -> int:
    """
    Remove duplicate comparison rows (same pair, same winner, same transitive depth) keeping only the newest entry.
    Uses a more relaxed matching than exact timestamps.
    
    Returns:
        Number of duplicates removed.
    """
    try:
        with get_db_connection() as conn:
            cur = conn.execute(
                """
                DELETE FROM comparisons
                WHERE id NOT IN (
                    SELECT MAX(id)
                    FROM comparisons
                    GROUP BY
                        CASE
                            WHEN filename_a < filename_b THEN filename_a
                            ELSE filename_b
                        END,
                        CASE
                            WHEN filename_a < filename_b THEN filename_b
                            ELSE filename_a
                        END,
                        winner,
                        transitive_depth
                )
                """
            )
            conn.commit()
            return max(cur.rowcount, 0)
    except Exception as e:
        logger.debug(f"Failed to deduplicate comparisons: {e}")
        return 0


def delete_comparisons_for_image(filename: str) -> int:
    """Delete all comparisons involving a specific image.

    Returns:
        Number of rows deleted.
    """
    try:
        with get_db_connection() as conn:
            cur = conn.execute(
                """
                DELETE FROM comparisons
                WHERE filename_a=? OR filename_b=?
                """,
                (filename, filename),
            )
            conn.commit()
            return cur.rowcount
    except Exception as e:
        logger.error(f"Error deleting comparisons for {filename}: {e}")
        return 0


def get_effective_comparison_count(filename: str) -> float:
    """Get the time-decayed sum of weights (impact factors) of all comparisons for an image.
    Newest comparisons have weight * 1.0, 10th newest has weight * 0.5.
    """
    try:
        with get_db_connection() as conn:
            rows = conn.execute(
                """
                SELECT weight FROM comparisons 
                WHERE filename_a=? OR filename_b=?
                ORDER BY timestamp DESC
                """,
                (filename, filename),
            ).fetchall()

            effective_count = 0.0
            for i, row in enumerate(rows):
                weight = float(row["weight"]) if row["weight"] is not None else 1.0
                decay = 0.5 ** (i / 9.0)
                effective_count += weight * decay
            return effective_count
    except Exception as e:
        logger.error(f"Error getting effective comparison count for {filename}: {e}")
        return 0.0
