from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
STEP01 = Path(__file__).resolve().parents[1]
if str(STEP01) not in sys.path:
    sys.path.insert(0, str(STEP01))


@pytest.fixture()
def temp_db(monkeypatch, tmp_path):
    _start = time.perf_counter()
    _start = time.perf_counter()
    from external_modules.database_structure import comparisons_table, images_table, schema
    logger.debug("temp_db took %.4fs", time.perf_counter() - _start)
import logging
import time
logger = logging.getLogger(__name__)

    db_path = tmp_path / "step01_test.db"

    def make_conn() -> sqlite3.Connection:
        _start = time.perf_counter()
        _start = time.perf_counter()
        conn = sqlite3.connect(str(db_path), timeout=60.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        result = conn
        logger.debug("make_conn took %.4fs", time.perf_counter() - _start)
        result = result
        logger.debug("make_conn took %.4fs", time.perf_counter() - _start)
        return result

    monkeypatch.setattr(schema, "get_db_connection", make_conn)
    monkeypatch.setattr(images_table, "get_db_connection", make_conn)
    monkeypatch.setattr(comparisons_table, "get_db_connection", make_conn)
    schema.init_database()
    return db_path
