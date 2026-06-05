from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Iterator

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ..logger import SharedLogger, get_logger
from .. import tasks


@pytest.fixture(autouse=True)
def reset_task_state() -> Iterator[None]:
    SharedLogger.clear_name_filters()
    with tasks._TASK_LOCK:
        tasks._TASK_OUTPUT.clear()
        tasks._TASK_CANCEL.clear()
    yield
    SharedLogger.clear_name_filters()
    with tasks._TASK_LOCK:
        tasks._TASK_OUTPUT.clear()
        tasks._TASK_CANCEL.clear()


def test_capture_stream_replaces_progress_lines_and_flushes() -> None:
    lines: list[str] = []
    stream = tasks._CaptureStream(lines, None)

    stream.write("starting\n10%\n20%\npartial")
    assert lines == ["starting", "20%"]

    stream._flush_remaining()
    assert lines == ["starting", "20%", "partial"]


def test_start_task_registers_running_task_and_invokes_thread_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[dict[str, object]] = []

    class FakeThread:
        def __init__(self, target, args, daemon) -> None:
            created.append({"target": target, "args": args, "daemon": daemon})

        def start(self) -> None:
            created.append({"started": True})

    monkeypatch.setattr(tasks.threading, "Thread", FakeThread)

    def run_task(task_id: str) -> None:
        tasks.set_task_output(task_id, {"status": "running", "ts": 123.0})

    task_id, response = tasks.start_task(run_task, task_prefix="demo", args=())

    assert task_id.startswith("demo_")
    assert response == {"task_id": task_id, "status": "started"}
    assert created[0]["daemon"] is True
    assert tasks.get_task_status(task_id, 0)["status"] == "running"


def test_run_captured_marks_done_and_preserves_logs() -> None:
    root_logger = logging.getLogger()
    original_level = root_logger.level
    root_logger.setLevel(logging.INFO)
    task_logger = get_logger("shared.tests.tasks.runner")
    task_id = "task_runner"
    tasks.set_task_output(task_id, {"status": "running", "ts": 1.0})

    def run_task(current_task_id: str) -> None:
        task_logger.info("hello from task")
        print("stdout line")
        assert current_task_id == task_id

    try:
        tasks._run_captured(task_id, run_task)
    finally:
        root_logger.setLevel(original_level)

    status = tasks.get_task_status(task_id, 0)
    assert status["status"] == "done"
    assert status["result"] == {"message": "completed"}
    assert "INFO shared.tests.tasks.runner - hello from task" in status["_log_new"]
    assert "stdout line" in status["_log_new"]


def test_cancel_task_short_circuits_run_captured() -> None:
    task_id = "task_cancelled"
    tasks.set_task_output(task_id, {"status": "running", "ts": 1.0})

    called = {"value": False}

    def run_task(current_task_id: str) -> None:
        called["value"] = True

    assert tasks.cancel_task(task_id) is True
    tasks._run_captured(task_id, run_task)

    status = tasks.get_task_status(task_id, 0)
    assert called["value"] is False
    assert status["status"] == "cancelled"
    assert "Task cancelled before start" in status["_log_new"]
    assert tasks.cancel_task(task_id) is False
