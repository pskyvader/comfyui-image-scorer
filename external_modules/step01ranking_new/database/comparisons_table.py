"""Comparisons table operations - full history of all comparisons."""

from typing import List, Dict, Any
from datetime import datetime, timedelta
from .schema import get_db_connection


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
    if winner not in (filename_a, filename_b):
        return 0

    try:
        with get_db_connection() as conn:
            # Check if exists
            exists = conn.execute(
                """
                SELECT id FROM comparisons 
                WHERE timestamp = ? AND 
                ((filename_a = ? AND filename_b = ?) OR (filename_a = ? AND filename_b = ?))
                """,
                (timestamp, filename_a, filename_b, filename_b, filename_a)
            ).fetchone()
            
            if exists:
                return exists['id']
                
            cur = conn.execute(
                """
                INSERT INTO comparisons(filename_a, filename_b, winner, timestamp, weight, transitive_depth)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (filename_a, filename_b, winner, timestamp, weight, transitive_depth),
            )
            conn.commit()
            return cur.lastrowid or 0
    except Exception:
        # Silently fail during rebuild/batch operations to avoid console spam.
        # Foreign key errors are expected if history contains deleted images.
        return 0


def add_comparison(
    filename_a: str,
    filename_b: str,
    winner: str,
    weight: float = 1.0,
    transitive_depth: int = 0,
) -> int:
    """
    Record a comparison result.

    Args:
        filename_a: First image filename
        filename_b: Second image filename
        winner: Winning image filename (must be one of the above)
        weight: Comparison weight (1.0 for direct, <1.0 for transitive)
        transitive_depth: Depth of transitive inference (0 for direct comparison)

    Returns:
        The insert ID if successful, 0 otherwise
    """
    if winner not in (filename_a, filename_b):
        print(f"Error: winner must be one of the compared images")
        return False

    try:
        with get_db_connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO comparisons(filename_a, filename_b, winner, timestamp, weight, transitive_depth)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP, ?, ?)
                """,
                (filename_a, filename_b, winner, weight, transitive_depth),
            )
            conn.commit()
            return cur.lastrowid or 0
    except Exception as e:
        print(f"Error adding comparison: {e}")
        return 0


def get_recent_comparisons(
    filename: str, days: int = 30, limit: int = 100
) -> List[Dict[str, Any]]:
    """
    Get recent comparisons for an image (within N days).

    Used for confidence calculation (stability over recent window).
    """
    try:
        cutoff_date = datetime.now() - timedelta(days=days)
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
        print(f"Error getting recent comparisons for {filename}: {e}")
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
        print(f"Error getting comparison count for {filename}: {e}")
        return 0


def get_total_comparisons() -> int:
    """Get total count of all comparisons in database."""
    try:
        with get_db_connection() as conn:
            row = conn.execute("SELECT COUNT(*) as cnt FROM comparisons").fetchone()
            return row["cnt"] if row else 0
    except Exception as e:
        print(f"Error getting total comparisons: {e}")
        return 0


def get_all_comparisons() -> List[Dict[str, Any]]:
    """Get all comparisons from the database."""
    try:
        with get_db_connection() as conn:
            rows = conn.execute("SELECT * FROM comparisons").fetchall()
            return [dict(row) for row in rows]
    except Exception as e:
        print(f"Error getting all comparisons: {e}")
        return []


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
        print(f"Error deleting comparisons for {filename}: {e}")
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
        print(f"Error getting effective comparison count for {filename}: {e}")
        return 0.0

