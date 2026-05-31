"""Background task infrastructure shared across sections."""

from __future__ import annotations

import io
import logging
import sys
import threading
import time
from contextlib import redirect_stdout, redirect_stderr
from typing import Any, Callable

from shared.logger import SharedLogger, TaskLogHandler, get_logger

logger = get_logger(__name__)

_TASK_OUTPUT: dict[str, dict[str, Any]] = {}
_TASK_LOCK = threading.Lock()
_TASK_CANCEL: set[str] = set()
_MAX_LOG_LINES = 500


class _CaptureStream(io.TextIOBase):
    def __init__(
        self, lines: list[str], original_stream: io.TextIOWrapper | None
    ) -> None:
        _start = time.perf_counter()
        self.lines = lines
        self._buf = ""
        self.original_stream = original_stream

    def write(self, s: str) -> int:
        _start = time.perf_counter()
        if self.original_stream:
            self.original_stream.write(s)
            self.original_stream.flush()

        self._buf += s

        while True:
            if "\r\n" in self._buf:
                line, self._buf = self._buf.split("\r\n", 1)
                self._process_line(line)
            elif "\r" in self._buf:
                line, self._buf = self._buf.split("\r", 1)
                self._process_line(line)
            elif "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                self._process_line(line)
            else:
                break

        result = len(s)

        return result

    def _is_progress_line(self, line: str) -> bool:
        _start = time.perf_counter()
        indicators = ["%", "|", "img/s", "items/s", "[00:", "it/s"]
        result = any(indicator in line for indicator in indicators)

        return result

    def _process_line(self, line: str) -> None:
        _start = time.perf_counter()
        if line:
            is_progress = self._is_progress_line(line)
            if is_progress and self.lines:
                if any(
                    indicator in self.lines[-1]
                    for indicator in ["%", "|", "img/s", "items/s", "[00:", "it/s"]
                ):
                    self.lines[-1] = line
                else:
                    self.lines.append(line)
            else:
                self.lines.append(line)
            if len(self.lines) > _MAX_LOG_LINES:
                self.lines.pop(0)

    def flush(self) -> None:
        _start = time.perf_counter()
        if self.original_stream:
            self.original_stream.flush()
        if self._buf:
            self._process_line(self._buf)
            self._buf = ""


def _run_captured(task_id: str, fn: Callable, *args, **kwargs) -> None:
    _start = time.perf_counter()
    lines: list[str] = []

    with _TASK_LOCK:
        info = _TASK_OUTPUT[task_id]
        info["_log"] = lines

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    cap_stdout = _CaptureStream(lines, original_stdout)
    cap_stderr = _CaptureStream(lines, original_stderr)

    root_logger = logging.getLogger()
    handler = TaskLogHandler(lines=lines, owner_thread_id=threading.get_ident())
    SharedLogger.register_task_buffer(task_id, lines)
    root_logger.addHandler(handler)

    try:
        with SharedLogger.task_context(task_id):
            with redirect_stdout(cap_stdout), redirect_stderr(cap_stderr):
                if task_id in _TASK_CANCEL:
                    lines.append("Task cancelled before start")
                else:
                    fn(task_id, *args, **kwargs)
                    with _TASK_LOCK:
                        info = _TASK_OUTPUT[task_id]
                        if info["status"] == "running":
                            info["status"] = "done"
                            info["result"] = {"message": "completed"}
    finally:
        root_logger.removeHandler(handler)
        SharedLogger.unregister_task_buffer(task_id)
        cap_stdout.flush()
        cap_stderr.flush()


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


def get_task_status(task_id: str, since: int) -> dict[str, Any]:
    _start = time.perf_counter()
    with _TASK_LOCK:
        info = _TASK_OUTPUT[task_id]
    resp = dict(info)
    log_lines = resp.pop("_log", None)
    if log_lines is not None:
        resp["_log_total"] = len(log_lines)
        resp["_log_new"] = log_lines[since:]

    return resp


def cancel_task(task_id: str) -> bool:
    _start = time.perf_counter()
    with _TASK_LOCK:
        info = _TASK_OUTPUT[task_id]
        if info["status"] in ("done", "error", "cancelled"):
            result = False
        else:
            _TASK_CANCEL.add(task_id)
            info["status"] = "cancelled"
            if "_log" in info:
                info["_log"].append("Task cancelled by user")
            else:
                info["_log"] = ["Task cancelled by user"]
            result = True

    return result
