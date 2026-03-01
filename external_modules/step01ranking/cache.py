import sqlite3
from typing import List, Dict, Any, Optional
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
                score_modifier INTEGER DEFAULT 0
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


def start_scan() -> None:
    """Mark scanning as started."""
    print("start scan")
    _set_meta("scanning", "1")
    _set_meta("finished", "0")


def finish_scan() -> None:
    """Mark scanning as finished."""
    print("finish scan")
    _set_meta("scanning", "0")
    _set_meta("finished", "1")


def is_scanning() -> bool:
    return _get_meta("scanning") == "1"


def is_finished() -> bool:
    return _get_meta("finished") == "1"


# ───────────────────────────
# cache operations
# ───────────────────────────
def add(
    path: str,
    score: int | None = None,
    comparison_count: int = 0,
    score_modifier: int = 0,
) -> None:
    """Add image path to cache with optional scoring metadata."""
    with _get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO cache(path, score, comparison_count, score_modifier) VALUES (?, ?, ?, ?)",
            (path, score, comparison_count, score_modifier),
        )
        conn.commit()


def in_cache(path: str) -> bool:
    with _get_conn() as conn:
        row = conn.execute("SELECT 1 FROM cache WHERE path=?", (path,)).fetchone()
        return bool(row)


def get(unscored_only: bool = True) -> List[str]:
    """Return cached image paths."""
    global _last_served
    _last_served = time()
    with _get_conn() as conn:
        if unscored_only:
            rows = conn.execute("SELECT path FROM cache WHERE score IS NULL").fetchall()
        else:
            rows = conn.execute("SELECT path FROM cache").fetchall()
        return [r["path"] for r in rows]


# def get_unscored_from_db() -> List[str]:
#     """Return only images with score=NULL (truly unscored)."""
#     with _get_conn() as conn:
#         rows = conn.execute("SELECT path FROM cache WHERE score IS NULL").fetchall()
#         return [r["path"] for r in rows]


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


def get_scored_not_compared(score: int = 0, limit: int = 10) -> List[str]:
    """Return images that are scored but not yet fully compared."""
    with _get_conn() as conn:
        query = "SELECT path FROM cache WHERE 1"

        if 0 < limit < 10:
            query += " AND comparison_count<" + str(limit)
        else:
            query += " AND comparison_count<10"

        if 1 <= score <= 5:
            query += " AND score=" + str(score)
        else:
            query += " AND score IS NOT NULL"
        rows = conn.execute(query).fetchall()
        result = [r["path"] for r in rows]
        print(
            f"[cache.get_scored_not_compared] Found {len(result)} scored not-compared images"
        )
        if result:
            for path in result[:3]:  # Show first 3
                print(f"  - {path}")
        return result


def get_all_scored() -> List[str]:
    """Return ALL images that have been scored (for diagnostics)."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT path, score, comparison_count FROM cache WHERE score IS NOT NULL"
        ).fetchall()
        print(f"[cache.get_all_scored] Found {len(rows)} scored images total")
        for row in rows[:5]:
            print(
                f"  - {row['path']} (scored={row['score']}, compared={row['comparison_count']})"
            )
        return [r["path"] for r in rows]


def get_cached_metadata(path: str) -> Optional[Dict[str, int | None]]:
    """Get cached score/comparison_count/score_modifier from database."""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT score, comparison_count, score_modifier FROM cache WHERE path=?",
            (path,),
        ).fetchone()
        if row:
            return {
                "score": row["score"],
                "comparison_count": row["comparison_count"],
                "score_modifier": row["score_modifier"],
            }
        return None


def update_cached_metadata(
    path: str,
    score: int | None = None,
    comparison_count: int | None = None,
    score_modifier: int | None = None,
) -> None:
    """Update cached score/comparison_count/score_modifier in database."""
    with _get_conn() as conn:
        updates: List[str] = []
        values: List[int | str] = []

        if score is not None:
            updates.append("score=?")
            values.append(score)
        if comparison_count is not None:
            updates.append("comparison_count=?")
            values.append(comparison_count)
        if score_modifier is not None:
            updates.append("score_modifier=?")
            values.append(score_modifier)

        if not updates:
            return

        values.append(path)
        query = f"UPDATE cache SET {', '.join(updates)} WHERE path=?"
        conn.execute(query, values)
        conn.commit()


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
