"""Background task infrastructure shared across sections."""

from __future__ import annotations

import io
import logging
import sys
import threading
import time
from contextlib import redirect_stdout, redirect_stderr
from typing import Any, Callable
logger = logging.getLogger(__name__)

_TASK_OUTPUT: dict[str, dict[str, Any]] = {}
_TASK_LOCK = threading.Lock()
_TASK_CANCEL: set[str] = set()
_MAX_LOG_LINES = 500


class _CaptureStream(io.TextIOBase):
    def __init__(
        _start = time.perf_counter()
        _start = time.perf_counter()
        self,
        lines: list[str],
        original_stream=None,
        logger.debug("__init__ took %.4fs", time.perf_counter() - _start)
    ) -> None:
        self.lines = lines
        self._buf = ""
        self.original_stream = original_stream

def write(str) -> int:
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
        logger.debug("write took %.4fs", time.perf_counter() - _start)
        return result

def _is_progress_line(str) -> bool:
        result = any(
        logger.debug("_is_progress_line took %.4fs", time.perf_counter() - _start)
        return result
            indicator in line
            for indicator in [
                "%",
                "|",
                "img/s",
                "items/s",
                "[00:",
                "it/s",
            ]
        )

def _process_line(str) -> None:
        if not line:
            logger.debug("_process_line took %.4fs", time.perf_counter() - _start)
            return

        is_progress = self._is_progress_line(line)

        if is_progress and self.lines:
            if any(
                indicator in self.lines[-1]
                for indicator in [
                    "%",
                    "|",
                    "img/s",
                    "items/s",
                    "[00:",
                    "it/s",
                ]
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
        _start = time.perf_counter()
        if self.original_stream:
            self.original_stream.flush()

        if self._buf:
            self._process_line(self._buf)
            self._buf = ""
            logger.debug("flush took %.4fs", time.perf_counter() - _start)


class _BufferHandler(logging.Handler):
    def __init__(
        _start = time.perf_counter()
        _start = time.perf_counter()
        self,
        lines: list[str],
        original_stdout=None,
        logger.debug("__init__ took %.4fs", time.perf_counter() - _start)
    ) -> None:
        super().__init__()

        self.lines = lines
        self.original_stdout = original_stdout or sys.stdout

def emit(logging.LogRecord) -> None:
        msg = self.format(record)

        self.lines.append(msg)

        if len(self.lines) > _MAX_LOG_LINES:
            self.lines.pop(0)

        self.original_stdout.write(msg + "\n")
        self.original_stdout.flush()


def _run_captured(
    _start = time.perf_counter()
    _start = time.perf_counter()
    task_id: str,
    fn: Callable,
    *args,
    **kwargs,
    logger.debug("_run_captured took %.4fs", time.perf_counter() - _start)
) -> None:
    lines: list[str] = []

    with _TASK_LOCK:
        info = _TASK_OUTPUT.get(task_id)

        if info:
            info["_log"] = lines

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    cap_stdout = _CaptureStream(
        lines,
        original_stdout,
    )

    cap_stderr = _CaptureStream(
        lines,
        original_stderr,
    )

    root_logger = logging.getLogger()

    handler = _BufferHandler(
        lines,
        original_stdout=original_stdout,
    )

    handler.setLevel(logging.INFO)

    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-5s %(name)s - %(message)s",
            datefmt="%H:%M:%S",
        )
    )

    root_logger.addHandler(handler)

    with redirect_stdout(cap_stdout), redirect_stderr(cap_stderr):
        if task_id in _TASK_CANCEL:
            lines.append("Task cancelled before start")
            return

        fn(task_id, *args, **kwargs)

        with _TASK_LOCK:
            info = _TASK_OUTPUT.get(task_id)

            if info and info.get("status") == "running":
                info["status"] = "done"
                info["result"] = {"message": "completed"}

    root_logger.removeHandler(handler)

    cap_stdout.flush()
    cap_stderr.flush()


def start_task(
    _start = time.perf_counter()
    _start = time.perf_counter()
    fn: Callable,
    *,
    task_prefix: str = "task",
    args: tuple = (),
    logger.debug("start_task took %.4fs", time.perf_counter() - _start)
) -> tuple[str, dict[str, Any]]:
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

    return task_id, {
        "task_id": task_id,
        "status": "started",
    }


def set_task_output(
    _start = time.perf_counter()
    _start = time.perf_counter()
    task_id: str,
    data: dict[str, Any],
    logger.debug("set_task_output took %.4fs", time.perf_counter() - _start)
) -> None:
    with _TASK_LOCK:
        _TASK_OUTPUT[task_id] = data


def get_task_status(
    _start = time.perf_counter()
    _start = time.perf_counter()
    task_id: str,
    since: int = 0,
    logger.debug("get_task_status took %.4fs", time.perf_counter() - _start)
) -> dict[str, Any] | None:
    with _TASK_LOCK:
        info = _TASK_OUTPUT.get(task_id)

    if info is None:
        return None

    resp = dict(info)

    log_lines = resp.pop("_log", None)

    if log_lines is not None:
        total = len(log_lines)

        resp["_log_total"] = total
        resp["_log_new"] = log_lines[since:]

    return resp


def cancel_task(str) -> bool:
    with _TASK_LOCK:
        info = _TASK_OUTPUT.get(task_id)

        if info is None:
            result = False
            logger.debug("cancel_task took %.4fs", time.perf_counter() - _start)
            return result

        if info.get("status") in (
            "done",
            "error",
            "cancelled",
        ):
            result = False
            logger.debug("cancel_task took %.4fs", time.perf_counter() - _start)
            return result

        _TASK_CANCEL.add(task_id)

        info["status"] = "cancelled"

        info["_log"] = info.get("_log", []) + ["Task cancelled by user"]

    result = True
    logger.debug("cancel_task took %.4fs", time.perf_counter() - _start)
    return result
