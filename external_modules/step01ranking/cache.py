import sqlite3
from typing import List, Dict, Any, Optional, Tuple
from time import time
from shared.paths import cache_file


# ───────────────────────────
# SQLite helpers
# ───────────────────────────
def _get_conn():
    conn = sqlite3.connect(cache_file)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _get_conn() as conn:
        # main cache table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache (
                path TEXT PRIMARY KEY,
                score INTEGER DEFAULT NULL,
                comparison_count INTEGER DEFAULT 0,
                score_modifier INTEGER DEFAULT 0,
                last_compared TEXT DEFAULT NULL
            )
            """
        )
        # meta table for scan flags and absolute total
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT
            )
            """
        )

        conn.commit()


def _set_meta(key: str, value: str):
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO meta(key,value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        conn.commit()


def _get_meta(key: str) -> str | None:
    with _get_conn() as conn:
        row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None


# ───────────────────────────
# lifecycle / scan flags
# ───────────────────────────


# 1️⃣ Get total elements per level, grouped by score
def get_total_per_level(score: int = 0) -> Dict[int, Dict[int, int]]:
    query = "SELECT score, comparison_count, COUNT(*) FROM cache "
    params = []

    if 1 <= score <= 5:
        query += "WHERE score = ? "
        params.append(score)
    else:
        query += "WHERE score IS NOT NULL "

    query += "GROUP BY score, comparison_count"

    with _get_conn() as connection:
        cursor = connection.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()

    result = {}
    for s, lvl, count in rows:
        if s not in result:
            result[s] = {}
        result[s][lvl] = count

    return result


# 2️⃣ Fetch paths for a specific tier
def get_images_by_level(score: int, level: int):
    query = "SELECT path FROM cache WHERE score = ? AND comparison_count = ?"
    with _get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(query, [score, level])
        return [row[0] for row in cursor.fetchall()]


# ───────────────────────────
# cache operations
# ───────────────────────────
def add(
    path: str,
    score: int | None = None,
    comparison_count: int = 0,
    score_modifier: int = 0,
    last_compared: str | None = None,
) -> None:
    """Add image path to cache with optional scoring metadata."""
    with _get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO cache(path, score, comparison_count, score_modifier,last_compared) VALUES (?, ?, ?, ?,?)",
            (path, score, comparison_count, score_modifier, last_compared),
        )
        conn.commit()


def get_all(unscored_only: bool = True) -> List[str]:
    """Return cached image paths."""
    global _last_served
    _last_served = time()
    with _get_conn() as conn:
        if unscored_only:
            rows = conn.execute("SELECT path FROM cache WHERE score IS NULL").fetchall()
        else:
            rows = conn.execute("SELECT path FROM cache").fetchall()
        return [r["path"] for r in rows]


def total_cached_unscored() -> int:
    """Count how many unscored images are in cache."""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM cache WHERE score IS NULL"
        ).fetchone()
        return row["cnt"]


def get_comparison_stats() -> Dict[str, int]:
    """Get stats for comparison mode."""
    with _get_conn() as conn:
        scored = conn.execute(
            "SELECT COUNT(*) as cnt FROM cache WHERE score IS NOT NULL"
        ).fetchone()["cnt"]
        not_compared = conn.execute(
            "SELECT COUNT(*) as cnt FROM cache WHERE score IS NOT NULL AND comparison_count<10"
        ).fetchone()["cnt"]
        fully_compared = conn.execute(
            "SELECT COUNT(*) as cnt FROM cache WHERE comparison_count>=10"
        ).fetchone()["cnt"]
        partially_compared = conn.execute(
            "SELECT COUNT(*) as cnt FROM cache WHERE comparison_count>0 AND comparison_count<10"
        ).fetchone()["cnt"]
        total = conn.execute("SELECT COUNT(*) as cnt FROM cache").fetchone()["cnt"]
        return {
            "scored": scored,
            "not_compared": not_compared,
            "fully_compared": fully_compared,
            "partially_compared": partially_compared,
            "total": total,
        }


def get_cached_metadata(path: str) -> Optional[Dict[str, int | float | str | None]]:
    """Get cached score/comparison_count/score_modifier from database."""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT score, comparison_count, score_modifier, last_compared FROM cache WHERE path=?",
            (path,),
        ).fetchone()
        if row:
            return {
                "score": int(row["score"]) if row["score"] else None,
                "comparison_count": int(row["comparison_count"]),
                "score_modifier": float(row["score_modifier"]),
                "last_compared": str(row["last_compared"]),
            }
        return None


_last_served: float = 0.0


def total_cached() -> int:
    with _get_conn() as conn:
        row = conn.execute("SELECT COUNT(*) as cnt FROM cache").fetchone()
        return row["cnt"]


# ───────────────────────────
# totals
# ───────────────────────────
def set_absolute_total(n: int) -> None:
    _set_meta("absolute_total", str(n))


def get_absolute_total() -> int:
    val = _get_meta("absolute_total")
    return int(val) if val else 0


# ───────────────────────────
# Initialize DB on import
# ───────────────────────────
init_db()
