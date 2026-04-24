"""Database schema definitions and connection management."""

import sqlite3
from pathlib import Path
from typing import Optional
from shared.paths import cache_file


def get_db_connection() -> sqlite3.Connection:
    """
    Create and return SQLite connection with row factory.
    Reusable pattern from original cache.py
    """
    db_path = Path(cache_file)
    # Increased timeout to 60s for background workers
    conn = sqlite3.connect(str(db_path), timeout=60.0)
    conn.row_factory = sqlite3.Row
    
    # Enable performance and integrity features
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    
    return conn


def init_database() -> None:
    """
    Initialize database with all required tables.
    """
    with get_db_connection() as conn:
        # Meta table for system metadata
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT
            )
            """
        )

        # Images table - lightweight index (filename only, no paths)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS images (
                filename TEXT PRIMARY KEY,
                score REAL DEFAULT 0.5,
                confidence REAL DEFAULT 0.0,
                comparison_count INTEGER DEFAULT 0,
                last_compared_at TEXT,
                ranking_generation INTEGER DEFAULT 0
            )
            """
        )

        # Indexes for common queries
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_images_score ON images(score)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_images_confidence ON images(confidence)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_images_comparison_count ON images(comparison_count)
            """
        )

        # Comparisons table - full history (NOT separate from this table)
        conn.execute(
            """
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
            """
        )

        # Indexes for comparison queries
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_comparisons_timestamp ON comparisons(timestamp)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_comparisons_winner ON comparisons(winner)
            """
        )

        conn.commit()

    # Initialize metadata
    _set_meta_value("db_version", "2")
    _set_meta_value("ranking_generation", "0")


def _set_meta_value(key: str, value: str) -> None:
    """Store metadata value (internal use)."""
    with get_db_connection() as conn:
        conn.execute(
            "INSERT INTO meta(key,value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        conn.commit()


def get_meta_value(key: str) -> Optional[str]:
    """Retrieve metadata value."""
    with get_db_connection() as conn:
        row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None


# Initialize on import
init_database()
