from __future__ import annotations

import logging
import sqlite3
import sys
import time
from pathlib import Path
from typing import Iterator

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
STEP01 = Path(__file__).resolve().parents[1]
if str(STEP01) not in sys.path:
    sys.path.insert(0, str(STEP01))

from shared.logger import get_logger

logger = get_logger(__name__)


@pytest.fixture()
def temp_db(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    _start = time.perf_counter()
    from external_modules.database_structure import (
        comparisons_table,
        images_table,
        schema,
    )

    db_path = tmp_path / "step01_test.db"

    def make_conn() -> sqlite3.Connection:
        _inner_start = time.perf_counter()
        conn = sqlite3.connect(str(db_path), timeout=60.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    monkeypatch.setattr(schema, "get_db_connection", make_conn)
    monkeypatch.setattr(images_table, "get_db_connection", make_conn)
    monkeypatch.setattr(comparisons_table, "get_db_connection", make_conn)
    schema.init_database()
    yield db_path
