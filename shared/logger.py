"""Shared backend logging utilities."""

from __future__ import annotations

import sys
import io
import logging
import queue
import threading
import time
from contextlib import contextmanager
from collections.abc import Callable, Iterator
from typing import ClassVar, Literal, TextIO, overload

# ── Global tqdm tuning ────────────────────────────────────────────────
try:
    import tqdm as _tqdm_module

    _tqdm_module.tqdm.mininterval = 1.0
except ImportError:
    pass

import os

# Create the package-level logger immediately so that all child loggers
# created during module import have the correct parent chain instead of
# falling back to root.  Set to DEBUG so early import-time log calls pass
# isEnabledFor(); configure_package_logging() will pin the final level later.
logging.getLogger("comfyui_image_scorer").setLevel(logging.DEBUG)


LogLevelName = Literal["debug", "info", "warning", "error", "critical"]


def _custom_find_caller(
    self: object,
    stack_info: bool = False,
    stacklevel: int = 1,
) -> tuple[str, int, str, str | None]:
    try:
        f = sys._getframe(1)  # pyright: ignore[reportPrivateUsage]
    except ValueError:
        f = None
    while f is not None and getattr(f, "f_code", None):
        co = f.f_code
        filename = os.path.normcase(co.co_filename)
        if (
            "logger.py" in filename or filename == logging._srcfile
        ):  # pyright: ignore[reportPrivateUsage]
            f = f.f_back
        else:
            break
    if f is None:
        return "(unknown file)", 0, "(unknown function)", None

    co = f.f_code
    sinfo = None
    if stack_info:
        import traceback

        sio = io.StringIO()
        sio.write("Stack (most recent call last):\n")
        traceback.print_stack(f, file=sio)
        sinfo = sio.getvalue()
        if sinfo[-1] == "\n":
            sinfo = sinfo[:-1]
        sio.close()
    return co.co_filename, f.f_lineno, co.co_name, sinfo


logging.Logger.findCaller = _custom_find_caller

_PROGRESS_INDICATORS = ["%", "|", "img/s", "items/s", "[00:", "it/s"]


def _is_progress_line(line: str) -> bool:
    return any(indicator in line for indicator in _PROGRESS_INDICATORS)


# ── Global filter hook — called for EVERY output line ─────────────────

_log_filter_hook: Callable[[str, str | None], bool] | None = None


def set_log_filter_hook(fn: Callable[[str, str | None], bool] | None) -> None:
    """Install a hook that is called for **every** output line across all
    channels (console, task buffer, SSE stream).

    The hook receives ``(line, module_name)`` where *module_name* is
    ``None`` for raw I/O (print, tqdm, cancel messages).  Return
    ``True`` to allow the line or ``False`` to suppress it entirely.
    """
    global _log_filter_hook
    _log_filter_hook = fn


# ── The single output manager for ALL task output ─────────────────────


class _TaskOutput:
    """Single point of control for ALL task output: logs, progress, prints,
    SSE streaming. Everything routes through here."""

    _task_buffers: ClassVar[dict[str, list[str]]] = {}
    _lock: ClassVar[threading.RLock] = threading.RLock()
    _context: ClassVar[threading.local] = threading.local()
    MAX_LINES: ClassVar[int] = 500

    # ── Context ───────────────────────────────────────────────────────

    @classmethod
    @contextmanager
    def context(cls, task_id: str) -> Iterator[None]:
        previous = getattr(cls._context, "task_id", None)
        cls._context.task_id = task_id
        try:
            yield
        finally:
            if previous is None:
                delattr(cls._context, "task_id")
            else:
                cls._context.task_id = previous

    @classmethod
    def current_task_id(cls) -> str | None:
        return getattr(cls._context, "task_id", None)

    @classmethod
    def register_buffer(cls, task_id: str, lines: list[str]) -> None:
        with cls._lock:
            cls._task_buffers[task_id] = lines

    @classmethod
    def unregister_buffer(cls, task_id: str) -> None:
        with cls._lock:
            cls._task_buffers.pop(task_id, None)

    @classmethod
    def has_buffer(cls, task_id: str) -> bool:
        with cls._lock:
            return task_id in cls._task_buffers

    # ── The ONE output method ─────────────────────────────────────────

    @classmethod
    def write(
        cls,
        task_id: str | None,
        line: str,
        *,
        is_progress: bool = False,
        module_name: str | None = None,
    ) -> None:
        """Write a line of output. The single path for buffer + SSE.

        - is_progress=True : appended to buffer normally but **not**
          broadcast via SSE.
        - module_name     : when provided, checked against the naming
          filter (``SharedLogger.should_emit``).  Lines from filtered-out
          modules are silently dropped from the buffer and SSE stream
          entirely — not just from the console.
        """
        if task_id is None:
            return
        if _log_filter_hook is not None and not _log_filter_hook(line, module_name):
            return
        if module_name is not None and not SharedLogger.should_emit(module_name):
            return
        with cls._lock:
            lines = cls._task_buffers.get(task_id)
            if lines is None:
                return
            lines.append(line)
            while len(lines) > cls.MAX_LINES:
                lines.pop(0)
        if not is_progress:
            SSELogBroadcaster.broadcast(line)


# ── I/O capture (feeds into _TaskOutput) ─────────────────────────────


class CaptureStream(io.TextIOBase):
    """Wraps stdout/stderr during a task.

    - Passes all data through to the original stream (console).
    - Feeds completed lines into ``_TaskOutput``, the single output
      manager (buffer + SSE).
    - Progress lines (tqdm, etc.) are marked so ``_TaskOutput`` can
      replace the previous progress line instead of appending.
    - Standalone ``\\r`` (carriage return without ``\\n``) overwrites
      the internal buffer instead of creating a new line, preventing
      character‑by‑character output from flooding the system.
    """

    def __init__(
        self,
        lines: list[str],
        original_stream: TextIO | None,
        *,
        task_id: str | None = None,
    ) -> None:
        self.lines = lines
        self._buf = ""
        self.original_stream = original_stream
        self._task_id = task_id

    def write(self, s: str) -> int:
        if self.original_stream:
            self.original_stream.write(s)
            self.original_stream.flush()

        self._buf += s

        while True:
            if "\r\n" in self._buf:
                line, self._buf = self._buf.split("\r\n", 1)
                self._process_line(line)
            elif "\r" in self._buf:
                line, self._buf = self._buf.rsplit("\r", 1)
                self._process_line(line)
            elif "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                self._process_line(line)
            else:
                break

        result = len(s)

        return result

    def _process_line(self, line: str) -> None:
        if not line:
            return

        is_progress = _is_progress_line(line)
        task_id = self._task_id or _TaskOutput.current_task_id()
        if task_id is not None:
            if SharedLogger.frontend_enabled:
                _TaskOutput.write(task_id, line, is_progress=is_progress)
            return

        if is_progress and self.lines:
            last = self.lines[-1]
            if _is_progress_line(last):
                self.lines[-1] = line
                return
        self.lines.append(line)
        if len(self.lines) > _TaskOutput.MAX_LINES:
            self.lines.pop(0)

    def flush(self) -> None:
        if self.original_stream:
            self.original_stream.flush()

    def _flush_remaining(self) -> None:
        if self._buf:
            self._process_line(self._buf)
            self._buf = ""


# ── SSE broadcaster (used by _TaskOutput) ─────────────────────────────


class SSELogBroadcaster:
    """Broadcasts log lines to all connected SSE clients in real time.

    Uses a background dispatch thread with batching to avoid blocking
    log-producing threads under high throughput.
    """

    _subscribers: ClassVar[dict[int, queue.Queue[str]]] = {}
    _lock: ClassVar[threading.Lock] = threading.Lock()
    _counter: ClassVar[int] = 0
    _inbox: ClassVar[queue.Queue[str]] = queue.Queue(maxsize=5000)
    _dispatch_thread: ClassVar[threading.Thread | None] = None
    _dispatch_started: ClassVar[bool] = False

    @classmethod
    def _ensure_dispatch(cls) -> None:
        if cls._dispatch_started:
            return
        cls._dispatch_started = True
        cls._dispatch_thread = threading.Thread(target=cls._dispatch_loop, daemon=True)
        cls._dispatch_thread.start()

    @classmethod
    def _dispatch_loop(cls) -> None:
        BATCH_SIZE = 50
        BATCH_TIMEOUT = 0.1
        while True:
            batch: list[str] = []
            try:
                batch.append(cls._inbox.get(timeout=BATCH_TIMEOUT))
                for _ in range(BATCH_SIZE - 1):
                    try:
                        batch.append(cls._inbox.get_nowait())
                    except queue.Empty:
                        break
            except queue.Empty:
                continue

            with cls._lock:
                if not cls._subscribers:
                    continue
                dead: list[int] = []
                for sub_id, q in cls._subscribers.items():
                    for line in batch:
                        try:
                            q.put_nowait(line)
                        except queue.Full:
                            dead.append(sub_id)
                            break
                for sub_id in dead:
                    cls._subscribers.pop(sub_id, None)

    @classmethod
    def subscribe(cls) -> tuple[int, queue.Queue[str]]:
        cls._ensure_dispatch()
        with cls._lock:
            cls._counter += 1
            sub_id = cls._counter
            q: queue.Queue[str] = queue.Queue(maxsize=1000)
            cls._subscribers[sub_id] = q
        return sub_id, q

    @classmethod
    def unsubscribe(cls, sub_id: int) -> None:
        with cls._lock:
            cls._subscribers.pop(sub_id, None)

    @classmethod
    def broadcast(cls, line: str) -> None:
        cls._ensure_dispatch()
        try:
            cls._inbox.put_nowait(line)
        except queue.Full:
            pass


# ── Python logging integration ────────────────────────────────────────


class _DynamicModuleFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return SharedLogger.should_emit(record.name)


class TaskLogHandler(logging.Handler):
    """Capture unmanaged logging records for a single task thread.

    For log records that bypass SharedLogger (direct logging.getLogger()
    calls), this handler formats them and feeds them through _TaskOutput
    so they end up in the task buffer and SSE stream.
    """

    def __init__(self, lines: list[str], owner_thread_id: int) -> None:
        super().__init__(level=logging.DEBUG)
        self._lines = lines
        self._owner_thread_id = owner_thread_id

    def emit(self, record: logging.LogRecord) -> None:
        if record.thread != self._owner_thread_id:
            return
        if getattr(record, "_shared_logger_managed", False):
            return
        if not SharedLogger.should_emit(record.name):
            return
        if record.levelno < SharedLogger.frontend_level:
            return

        if not SharedLogger.frontend_enabled:
            return
        task_line = SharedLogger.format_task_line(
            module_name=record.name,
            level_name=record.levelname,
            message=record.getMessage(),
        )
        task_id = getattr(record, "task_id", None) or _TaskOutput.current_task_id()
        _TaskOutput.write(task_id, task_line, module_name=record.name)


class ModuleLogger:
    def __init__(self, module_name: str) -> None:
        self.module_name = module_name

    @property
    def _underlying(self) -> logging.Logger:
        return logging.getLogger(self.module_name)

    @property
    def level(self) -> int:
        return self._underlying.level

    @level.setter
    def level(self, value: int) -> None:
        self._underlying.level = value

    def setLevel(self, level: int) -> None:
        self._underlying.setLevel(level)

    def addHandler(self, hdlr: logging.Handler) -> None:
        self._underlying.addHandler(hdlr)

    def removeHandler(self, hdlr: logging.Handler) -> None:
        self._underlying.removeHandler(hdlr)

    def log(
        self,
        level_name: LogLevelName,
        message: str,
        *args: object,
        start_timer: float | None = None,
    ) -> None:
        # print(f"DEBUG_ML: level_name={level_name!r} message={message!r} args={args!r} start_timer={start_timer!r}")
        if args:
            message = message % args
        # print(f"DEBUG_ML: calling SharedLogger.log(module_name={self.module_name!r}, level_name={level_name!r}, message={message!r})")
        SharedLogger.log(
            module_name=self.module_name,
            level_name=level_name,
            message=message,
            start_timer=start_timer,
        )

    def debug(
        self, message: str, *args: object, start_timer: float | None = None
    ) -> None:
        self.log("debug", message, *args, start_timer=start_timer)

    def info(
        self, message: str, *args: object, start_timer: float | None = None
    ) -> None:
        self.log("info", message, *args, start_timer=start_timer)

    def warning(
        self, message: str, *args: object, start_timer: float | None = None
    ) -> None:
        self.log("warning", message, *args, start_timer=start_timer)

    def error(
        self, message: str, *args: object, start_timer: float | None = None
    ) -> None:
        self.log("error", message, *args, start_timer=start_timer)

    def exception(
        self, message: str, *args: object, start_timer: float | None = None
    ) -> None:
        self.log("error", message, *args, start_timer=start_timer)

    def critical(
        self, message: str, *args: object, start_timer: float | None = None
    ) -> None:
        self.log("critical", message, *args, start_timer=start_timer)


class SharedLogger:
    """Centralized backend logger and task log router.

    Filtering (name/level), formatting, and console output live here.
    Task buffer and SSE broadcast are delegated to ``_TaskOutput``,
    the single output manager.
    """

    frontend_enabled: ClassVar[bool] = False
    frontend_level: ClassVar[int] = logging.INFO
    allowed_exact_names: ClassVar[frozenset[str]] = frozenset()
    allowed_prefixes: ClassVar[tuple[str, ...]] = ()
    _name_filter: ClassVar[_DynamicModuleFilter] = _DynamicModuleFilter()

    @classmethod
    def install_root_filter(cls) -> None:
        root_logger = logging.getLogger()
        if cls._name_filter not in root_logger.filters:
            root_logger.addFilter(cls._name_filter)

    @classmethod
    def set_name_filters(
        cls,
        exact_names: set[str] | frozenset[str] | tuple[str, ...] | list[str],
        prefixes: tuple[str, ...] | list[str],
    ) -> None:
        cls.allowed_exact_names = frozenset(exact_names)
        cls.allowed_prefixes = tuple(prefixes)

    @classmethod
    def clear_name_filters(cls) -> None:
        cls.allowed_exact_names = frozenset()
        cls.allowed_prefixes = ()

    @classmethod
    def set_frontend_enabled(cls, enabled: bool) -> None:
        cls.frontend_enabled = enabled

    @classmethod
    def set_frontend_level(cls, level_name: LogLevelName) -> None:
        cls.frontend_level = cls._normalize_level(level_name)

    @classmethod
    def should_emit(cls, module_name: str) -> bool:
        if not cls.allowed_exact_names and not cls.allowed_prefixes:
            return True
        if module_name in cls.allowed_exact_names:
            return True
        result = any(module_name.startswith(prefix) for prefix in cls.allowed_prefixes)
        return result

    @classmethod
    def get_logger(cls, module_name: str) -> ModuleLogger:
        cls.install_root_filter()
        result = ModuleLogger(module_name)
        return result

    # ── Delegation to _TaskOutput ─────────────────────────────────────
    # Kept as classmethods so existing callers (tasks.py, tests) still
    # work without import changes.

    @classmethod
    def register_task_buffer(cls, task_id: str, lines: list[str]) -> None:
        _TaskOutput.register_buffer(task_id, lines)

    @classmethod
    def unregister_task_buffer(cls, task_id: str) -> None:
        _TaskOutput.unregister_buffer(task_id)

    @classmethod
    @contextmanager
    def task_context(cls, task_id: str) -> Iterator[None]:
        with _TaskOutput.context(task_id):
            yield

    @classmethod
    def current_task_id(cls) -> str | None:
        return _TaskOutput.current_task_id()

    # ── Formatting ────────────────────────────────────────────────────

    @classmethod
    def format_message(cls, message: str, start_timer: float | None) -> str:
        result = message

        caller_name = sys._getframe(
            4
        ).f_code.co_name  # pyright: ignore[reportPrivateUsage]

        if start_timer is not None:
            result = (
                f"{message} ({caller_name}) ({time.perf_counter() - start_timer:.4f}s)"
            )
        return result

    @classmethod
    def format_task_line(
        cls,
        module_name: str,
        level_name: str,
        message: str,
    ) -> str:
        result = f"{level_name.upper()} {module_name} - {message}"
        return result

    # ── The log method ────────────────────────────────────────────────

    @classmethod
    def log(
        cls,
        module_name: str,
        level_name: LogLevelName,
        message: str,
        start_timer: float | None,
        task_id: str | None = None,
    ) -> None:
        cls.install_root_filter()
        if not cls.should_emit(module_name):
            # print(f"mesage should not emit:{message[:10]}...", flush=True)
            return

        rendered_message = cls.format_message(message, start_timer)
        level = cls._normalize_level(level_name)
        _logger = logging.getLogger(module_name)
        # print(
        #     f"Message emit: {module_name} {level} {rendered_message[:10]}...",
        #     flush=True,
        # )
        # print(
        #     f"Message logger: {_logger}, enabled: {_logger.isEnabledFor(level)}",
        #     flush=True,
        # )
        _logger.log(
            level,
            rendered_message,
            extra={"_shared_logger_managed": True},
        )

        if cls.frontend_enabled:
            tid = task_id or cls.current_task_id()
            if tid:
                _TaskOutput.write(tid, rendered_message, module_name=module_name)

    @staticmethod
    def _normalize_level(level_name: LogLevelName) -> int:
        level_map: dict[LogLevelName, int] = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "critical": logging.CRITICAL,
        }
        result = level_map[level_name]
        return result


class CustomFormatter(logging.Formatter):
    """Custom formatter to trim level names, module names, function names, and messages."""

    def __init__(
        self,
        fmt: str | None = None,
        datefmt: str | None = None,
        trim_level_len: int | None = 3,
        trim_module_len: int | None = 15,
        trim_func_len: int | None = 15,
        trim_msg_len: int | None = None,
    ) -> None:
        super().__init__(fmt, datefmt)
        self.trim_level_len = trim_level_len
        self.trim_module_len = trim_module_len
        self.trim_func_len = trim_func_len
        self.trim_msg_len = trim_msg_len

    def format(self, record: logging.LogRecord) -> str:
        orig_levelname = record.levelname
        orig_name = record.name
        orig_funcName = record.funcName
        orig_msg = record.msg
        orig_args = record.args

        # Trim levelname
        if self.trim_level_len is not None:
            level_map = {
                "DEBUG": "DBG",
                "INFO": "INF",
                "WARNING": "WRN",
                "ERROR": "ERR",
                "CRITICAL": "CRT",
            }
            if self.trim_level_len == 3:
                record.levelname = level_map.get(orig_levelname, orig_levelname[:3])
            else:
                record.levelname = orig_levelname[: self.trim_level_len]

        # Trim module name to keep the last N characters (from the end)
        if self.trim_module_len is not None and orig_name:
            if len(orig_name) > self.trim_module_len:
                if self.trim_module_len > 3:
                    record.name = "..." + orig_name[-(self.trim_module_len - 3) :]
                else:
                    record.name = orig_name[-self.trim_module_len :]

        # Trim function name to keep the last N characters (from the end)
        if self.trim_func_len is not None and orig_funcName:
            if len(orig_funcName) > self.trim_func_len:
                if self.trim_func_len > 3:
                    record.funcName = "..." + orig_funcName[-(self.trim_func_len - 3) :]
                else:
                    record.funcName = orig_funcName[-self.trim_func_len :]

        # Trim message length
        if self.trim_msg_len is not None:
            msg = record.getMessage()
            if len(msg) > self.trim_msg_len:
                if self.trim_msg_len > 3:
                    record.msg = msg[: self.trim_msg_len - 3] + "..."
                else:
                    record.msg = msg[: self.trim_msg_len]
                record.args = ()

        try:
            result = super().format(record)
        finally:
            # Restore original values so other handlers/formatters are unaffected
            record.levelname = orig_levelname
            record.name = orig_name
            record.funcName = orig_funcName
            record.msg = orig_msg
            record.args = orig_args

        return result


@overload
def get_logger(module_name: None = None) -> logging.Logger: ...


@overload
def get_logger(module_name: str) -> ModuleLogger: ...


def get_logger(module_name: str | None = None) -> logging.Logger | ModuleLogger:
    if module_name is None:
        return logging.getLogger()
    result: ModuleLogger = SharedLogger.get_logger(module_name)
    return result


def configure_package_logging(
    level: int = logging.INFO,
    fmt: str | None = None,
    *,
    datefmt: str | None = "%H:%M:%S",
    trim_level_len: int | None = 3,
    trim_module_len: int | None = 15,
    trim_func_len: int | None = 15,
    trim_msg_len: int | None = None,
) -> None:
    if fmt is None:
        fmt = "[%(levelname)s] [%(name)s] [%(funcName)s] %(asctime)s %(message)s"

    logging.basicConfig(
        # level=level,  # intentionally not set — avoids changing external library log levels
        format=fmt,
        datefmt=datefmt,
    )

    # Apply CustomFormatter to all root handlers to handle the trimming
    formatter = CustomFormatter(
        fmt,
        datefmt=datefmt,
        trim_level_len=trim_level_len,
        trim_module_len=trim_module_len,
        trim_func_len=trim_func_len,
        trim_msg_len=trim_msg_len,
    )
    for handler in logging.root.handlers:
        handler.setFormatter(formatter)

    logging.getLogger("__main__").setLevel(level)

    pkg_logger = logging.getLogger("comfyui_image_scorer")
    pkg_logger.setLevel(level)
    # Rewire parent links in case any child loggers were created before
    # the package logger existed, then clear level caches.
    logging.root.manager._fixupParents(pkg_logger)
    _cleared = 0
    for _log_name, _log in list(logging.root.manager.loggerDict.items()):
        if isinstance(_log, logging.Logger) and _log_name.startswith(
            "comfyui_image_scorer"
        ):
            _log._cache.clear()
            _cleared += 1
    print(f"cleared:{_cleared}")

    # Suppress verbose logging from noisy external libraries
    # for logger_name in ["mediapipe", "PIL", "matplotlib", "urllib3", "onnxruntime"]:
    #     logging.getLogger(logger_name).setLevel(logging.WARNING)

    # Frontend logging (task buffer + SSE) disabled by default.
    # Set to True to also route logs to frontend clients.
    SharedLogger.set_frontend_enabled(False)


def log_message(
    module_name: str,
    level_name: LogLevelName,
    message: str,
    start_timer: float | None,
    task_id: str | None = None,
) -> None:
    SharedLogger.log(
        module_name=module_name,
        level_name=level_name,
        message=message,
        start_timer=start_timer,
        task_id=task_id,
    )


configure_package_logging()
