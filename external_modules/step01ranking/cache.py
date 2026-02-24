import sqlite3
from pathlib import Path
from typing import List
from time import time

DB_PATH = Path(__file__).parent / "cache.db"

# ───────────────────────────
# SQLite helpers
# ───────────────────────────
def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _get_conn() as conn:
        # main cache table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache (
                path TEXT PRIMARY KEY,
                valid INTEGER
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
def reset() -> None:
    """Clear the cache and reset scanning state."""
    with _get_conn() as conn:
        conn.execute("DELETE FROM cache")
        conn.execute("DELETE FROM meta")
        conn.commit()


def start_scan() -> None:
    """Mark scanning as started."""
    _set_meta("scanning", "1")
    _set_meta("finished", "0")


def finish_scan() -> None:
    """Mark scanning as finished."""
    _set_meta("scanning", "0")
    _set_meta("finished", "1")


def is_scanning() -> bool:
    return _get_meta("scanning") == "1"


def is_finished() -> bool:
    return _get_meta("finished") == "1"


# ───────────────────────────
# cache operations
# ───────────────────────────
def add(path: str) -> None:
    """Add image path to cache (enabled)."""
    with _get_conn() as conn:
        conn.execute("INSERT OR REPLACE INTO cache(path,valid) VALUES (?,1)", (path,))
        conn.commit()


def disable(path: str) -> None:
    """Mark an image as disabled (scored)."""
    with _get_conn() as conn:
        conn.execute("UPDATE cache SET valid=0 WHERE path=?", (path,))
        conn.commit()


def in_cache(path: str) -> bool:
    with _get_conn() as conn:
        row = conn.execute("SELECT 1 FROM cache WHERE path=?", (path,)).fetchone()
        return bool(row)


def get(valid_only: bool = True) -> List[str]:
    """Return cached image paths."""
    global _last_served
    _last_served = time()
    with _get_conn() as conn:
        if valid_only:
            rows = conn.execute("SELECT path FROM cache WHERE valid=1").fetchall()
        else:
            rows = conn.execute("SELECT path FROM cache").fetchall()
        return [r["path"] for r in rows]


_last_served: float = 0.0


def fast_serve() -> bool:
    """Return True if the cache was served recently (last second)."""
    return _last_served > time() - 1


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
