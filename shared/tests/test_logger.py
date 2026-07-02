from __future__ import annotations

import logging
import sys
import threading
import time
from pathlib import Path
from typing import Iterator

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ..logger import SharedLogger, TaskLogHandler, get_logger, log_message
from ..tasks import get_task_status, start_task


@pytest.fixture(autouse=True)
def reset_shared_logger_state() -> Iterator[None]:
    SharedLogger.clear_name_filters()
    SharedLogger.set_frontend_level("info")
    yield
    SharedLogger.clear_name_filters()
    SharedLogger.set_frontend_level("info")


def test_log_message_formats_elapsed_time(caplog: pytest.LogCaptureFixture) -> None:
    _start = time.perf_counter()

    with caplog.at_level(logging.DEBUG):
        log_message("shared.tests.logger.format", "debug", "formatted message", _start)

    assert len(caplog.records) == 1
    assert caplog.records[0].name == "shared.tests.logger.format"
    assert caplog.records[0].message.startswith("formatted message (")
    assert caplog.records[0].message.endswith("s)")


def test_name_filter_blocks_nonmatching_modules(
    caplog: pytest.LogCaptureFixture,
) -> None:
    SharedLogger.set_name_filters(
        exact_names=("shared.tests.logger.allowed",),
        prefixes=(),
    )

    with caplog.at_level(logging.INFO):
        log_message("shared.tests.logger.blocked", "info", "blocked", None)
        log_message("shared.tests.logger.allowed", "info", "allowed", None)

    assert [record.message for record in caplog.records] == ["allowed"]


def test_task_log_handler_captures_managed_and_legacy_records_once() -> None:
    lines: list[str] = []
    task_id = "logger_test_task"
    handler = TaskLogHandler(lines=lines, owner_thread_id=threading.get_ident())
    original_level = get_logger().level
    get_logger().setLevel(logging.INFO)
    get_logger().addHandler(handler)

    try:
        SharedLogger.register_task_buffer(task_id, lines)
        with SharedLogger.task_context(task_id):
            log_message("shared.tests.logger.managed", "info", "managed", None)
            logging.getLogger("shared.tests.logger.legacy").info("legacy")
    finally:
        get_logger().removeHandler(handler)
        get_logger().setLevel(original_level)
        SharedLogger.unregister_task_buffer(task_id)

    assert lines == [
        "INFO shared.tests.logger.managed - managed",
        "INFO shared.tests.logger.legacy - legacy",
    ]


def test_start_task_exposes_shared_logger_lines() -> None:
    original_level = get_logger().level
    get_logger().setLevel(logging.INFO)
    task_logger = get_logger("shared.tests.logger.task")

    def run_task(task_id: str) -> None:
        task_logger.info("task log line")
        logging.getLogger("shared.tests.logger.task.legacy").info("legacy task line")
        print("stdout line")

    try:
        task_id, _ = start_task(run_task, task_prefix="logger", args=())

        status: dict[str, object] = {}
        for _ in range(100):
            status = get_task_status(task_id, 0)
            if status["status"] == "done":
                break
            time.sleep(0.02)
    finally:
        get_logger().setLevel(original_level)

    assert status["status"] == "done"
    log_lines = status["_log_new"]
    assert "INFO shared.tests.logger.task - task log line" in log_lines
    assert "INFO shared.tests.logger.task.legacy - legacy task line" in log_lines
    assert "stdout line" in log_lines
