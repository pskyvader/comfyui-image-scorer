"""Database schema definitions and connection management."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from shared.paths import cache_file
import logging
import time

logger = logging.getLogger(__name__)

MU0 = 25.0
SIGMA0 = MU0 / 3.0


def get_db_connection() -> sqlite3.Connection:
    """Create and return SQLite connection with row factory."""

    db_path = Path(cache_file)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=60.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA mmap_size = 268435456")
    conn.execute("PRAGMA cache_size = -8192")
    return conn


def _ensure_meta_table(conn: sqlite3.Connection) -> None:

    conn.execute("""
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        """)


def _ensure_images_table(conn: sqlite3.Connection) -> None:
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS images (
            filename TEXT PRIMARY KEY,
            score REAL DEFAULT 0.5,
            rating_mu REAL DEFAULT {MU0},
            rating_sigma REAL DEFAULT {SIGMA0},
            comparison_count INTEGER DEFAULT 0,
            last_compared_at TEXT,
            ranking_generation INTEGER DEFAULT 0,
            prompt_tags TEXT
        )
        """)

    existing = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(images)").fetchall()
        if row["name"]
    }
    if "rating_mu" not in existing:
        conn.execute(f"ALTER TABLE images ADD COLUMN rating_mu REAL DEFAULT {MU0}")
    if "rating_sigma" not in existing:
        conn.execute(
            f"ALTER TABLE images ADD COLUMN rating_sigma REAL DEFAULT {SIGMA0}"
        )

    conn.execute("CREATE INDEX IF NOT EXISTS idx_images_score ON images(score)")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_images_comparison_count ON images(comparison_count)"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_images_rating_mu ON images(rating_mu)")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_images_rating_sigma ON images(rating_sigma)"
    )


def _ensure_comparisons_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS comparisons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename_a TEXT NOT NULL,
            filename_b TEXT NOT NULL,
            winner TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            weight REAL DEFAULT 1.0,
            transitive_depth INTEGER DEFAULT 0,
            FOREIGN KEY (filename_a) REFERENCES images(filename),
            FOREIGN KEY (filename_b) REFERENCES images(filename),
            FOREIGN KEY (winner) REFERENCES images(filename)
        )
        """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_comparisons_timestamp ON comparisons(timestamp)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_comparisons_winner ON comparisons(winner)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_comparisons_pair ON comparisons(filename_a, filename_b)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_comparisons_winner_files ON comparisons(winner, filename_a, filename_b)"
    )


def init_database() -> None:
    """Initialize database with all required tables."""

    with get_db_connection() as conn:
        _ensure_meta_table(conn)
        _ensure_images_table(conn)
        _ensure_comparisons_table(conn)
        conn.commit()

    _set_meta_value("db_version", "4")
    _set_meta_value("ranking_generation", "0")

    _start = time.perf_counter()
    _start = time.perf_counter()


def _set_meta_value(key: str, value: str) -> None:

    with get_db_connection() as conn:
        conn.execute(
            "INSERT INTO meta(key,value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        conn.commit()


def get_meta_value(key: str) -> str | None:
    with get_db_connection() as conn:
        row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        result = row["value"] if row else None
        return result


init_database()
