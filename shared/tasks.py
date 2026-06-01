"""Background task infrastructure shared across sections."""

from __future__ import annotations

import logging
import sys
import threading
import time
from contextlib import redirect_stdout, redirect_stderr
from typing import Any, Callable

from shared.logger import (
    SharedLogger,
    TaskLogHandler,
    _CaptureStream,
    _TaskOutput,
    get_logger,
)

logger = get_logger(__name__)

_TASK_OUTPUT: dict[str, dict[str, Any]] = {}
_TASK_LOCK = threading.Lock()
_TASK_CANCEL: set[str] = set()


def _run_captured(task_id: str, fn: Callable, *args, **kwargs) -> None:
    _start = time.perf_counter()
    lines: list[str] = []

    with _TASK_LOCK:
        info = _TASK_OUTPUT[task_id]
        info["_log"] = lines

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    cap_stdout = _CaptureStream(lines, original_stdout, task_id=task_id)
    cap_stderr = _CaptureStream(lines, original_stderr, task_id=task_id)

    root_logger = logging.getLogger()
    handler = TaskLogHandler(lines=lines, owner_thread_id=threading.get_ident())
    _TaskOutput.register_buffer(task_id, lines)
    root_logger.addHandler(handler)

    try:
        with _TaskOutput.context(task_id):
            with redirect_stdout(cap_stdout), redirect_stderr(cap_stderr):
                if task_id in _TASK_CANCEL:
                    _TaskOutput.write(task_id, "Task cancelled before start")
                else:
                    fn(task_id, *args, **kwargs)
                    with _TASK_LOCK:
                        info = _TASK_OUTPUT[task_id]
                        if info["status"] == "running":
                            info["status"] = "done"
                            info["result"] = {"message": "completed"}
    finally:
        root_logger.removeHandler(handler)
        cap_stdout._flush_remaining()
        cap_stderr._flush_remaining()
        _TaskOutput.unregister_buffer(task_id)


def start_task(
    fn: Callable, *, task_prefix: str, args: tuple
) -> tuple[str, dict[str, Any]]:
    _start = time.perf_counter()
    task_id = f"{task_prefix}_{int(time.time())}"

    with _TASK_LOCK:
        _TASK_OUTPUT[task_id] = {
            "status": "running",
            "ts": time.time(),
        }

    threading.Thread(
        target=_run_captured,
        args=(task_id, fn, *args),
        daemon=True,
    ).start()

    result = (task_id, {"task_id": task_id, "status": "started"})

    return result


def set_task_output(task_id: str, data: dict[str, Any]) -> None:
    _start = time.perf_counter()
    with _TASK_LOCK:
        _TASK_OUTPUT[task_id] = data


def get_task_status(task_id: str, since: int) -> dict[str, Any] | None:
    _start = time.perf_counter()
    with _TASK_LOCK:
        info = _TASK_OUTPUT.get(task_id)
    if info is None:
        return None
    resp = dict(info)
    log_lines = resp.pop("_log", None)
    if log_lines is not None:
        resp["_log_total"] = len(log_lines)
        resp["_log_new"] = log_lines[since:]

    return resp


def cancel_task(task_id: str) -> bool:
    _start = time.perf_counter()
    with _TASK_LOCK:
        info = _TASK_OUTPUT.get(task_id)
        if info is None:
            return False
        if info["status"] in ("done", "error", "cancelled"):
            result = False
        else:
            _TASK_CANCEL.add(task_id)
            info["status"] = "cancelled"
            if "_log" in info:
                _TaskOutput.write(task_id, "Task cancelled by user")
            else:
                info["_log"] = ["Task cancelled by user"]
            result = True

    return result
