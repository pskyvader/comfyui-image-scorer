from __future__ import annotations

from .database_structure import comparisons_table, images_table, schema
import time

from ...shared.logger import get_logger, ModuleLogger
logger: ModuleLogger = get_logger(__name__)


def test_normalize_comparisons_keeps_latest_pair_result(temp_db):
    _start = time.perf_counter()
    _start = time.perf_counter()
    _start = time.perf_counter()
    images_table.add_image("a.png")
    images_table.add_image("b.png")

    comparisons_table.add_historical_comparison(
        "a.png",
        "b.png",
        "a.png",
        "2026-01-01T00:00:00+00:00",
    )
    comparisons_table.add_historical_comparison(
        "a.png",
        "b.png",
        "a.png",
        "2026-01-02T00:00:00+00:00",
    )
    comparisons_table.add_historical_comparison(
        "a.png",
        "b.png",
        "b.png",
        "2026-01-03T00:00:00+00:00",
    )

    stats = comparisons_table.normalize_comparisons()
    rows = comparisons_table.get_all_comparisons()

    assert stats["same_direction_duplicates_removed"] == 1
    assert stats["contradictions_removed"] == 1
    assert len(rows) == 1
    assert rows[0]["winner"] == "b.png"


def test_normalize_comparisons_removes_self_links_and_missing_nodes(temp_db):
    _start = time.perf_counter()
    _start = time.perf_counter()
    _start = time.perf_counter()
    images_table.add_image("a.png")
    images_table.add_image("b.png")
    comparisons_table.add_historical_comparison(
        "a.png",
        "a.png",
        "a.png",
        "2026-01-01T00:00:00+00:00",
    )

    with schema.get_db_connection() as conn:
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute(
            """
            INSERT INTO comparisons(filename_a, filename_b, winner, timestamp, weight, transitive_depth)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "a.png",
                "missing.png",
                "a.png",
                "2026-01-02T00:00:00+00:00",
                1.0,
                0,
            ),
        )
        conn.commit()
        conn.execute("PRAGMA foreign_keys = ON")

    stats = comparisons_table.normalize_comparisons()
    assert stats["self_links_removed"] == 1
    assert stats["missing_nodes_removed"] == 1
    assert comparisons_table.get_all_comparisons() == []
