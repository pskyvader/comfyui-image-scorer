"""Shared backend logging utilities."""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from typing import ClassVar, Iterator, Literal

LogLevelName = Literal["debug", "info", "warning", "error", "critical"]


class _DynamicModuleFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        result = SharedLogger.should_emit(record.name)
        return result


class TaskLogHandler(logging.Handler):
    """Capture unmanaged logging records for a single task thread."""

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

        message = record.getMessage()
        task_line = SharedLogger.format_task_line(
            module_name=record.name,
            level_name=record.levelname,
            message=message,
        )
        SharedLogger.append_task_line(self._lines, task_line)


class ModuleLogger:
    def __init__(self, module_name: str) -> None:
        self.module_name = module_name

    def log(
        self,
        level_name: LogLevelName,
        message: str,
        start_timer: float | None = None,
    ) -> None:
        SharedLogger.log(
            module_name=self.module_name,
            level_name=level_name,
            message=message,
            start_timer=start_timer,
        )

    def debug(self, message: str, start_timer: float | None = None) -> None:
        self.log("debug", message, start_timer)

    def info(self, message: str, start_timer: float | None = None) -> None:
        self.log("info", message, start_timer)

    def warning(self, message: str, start_timer: float | None = None) -> None:
        self.log("warning", message, start_timer)

    def error(self, message: str, start_timer: float | None = None) -> None:
        self.log("error", message, start_timer)

    def critical(self, message: str, start_timer: float | None = None) -> None:
        self.log("critical", message, start_timer)


class SharedLogger:
    """Centralized backend logger and task log router."""

    frontend_level: ClassVar[int] = logging.INFO
    max_task_lines: ClassVar[int] = 500
    allowed_exact_names: ClassVar[frozenset[str]] = frozenset()
    allowed_prefixes: ClassVar[tuple[str, ...]] = ()
    _name_filter: ClassVar[_DynamicModuleFilter] = _DynamicModuleFilter()
    _filter_installed: ClassVar[bool] = False
    _task_buffers: ClassVar[dict[str, list[str]]] = {}
    _task_lock: ClassVar[threading.RLock] = threading.RLock()
    _task_context: ClassVar[threading.local] = threading.local()

    @classmethod
    def install_root_filter(cls) -> None:
        root_logger = logging.getLogger()
        if cls._name_filter not in root_logger.filters:
            root_logger.addFilter(cls._name_filter)
        cls._filter_installed = True

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

    @classmethod
    def register_task_buffer(cls, task_id: str, lines: list[str]) -> None:
        with cls._task_lock:
            cls._task_buffers[task_id] = lines

    @classmethod
    def unregister_task_buffer(cls, task_id: str) -> None:
        with cls._task_lock:
            if task_id in cls._task_buffers:
                del cls._task_buffers[task_id]

    @classmethod
    @contextmanager
    def task_context(cls, task_id: str) -> Iterator[None]:
        previous_task_id = getattr(cls._task_context, "task_id", None)
        cls._task_context.task_id = task_id
        try:
            yield
        finally:
            if previous_task_id is None:
                delattr(cls._task_context, "task_id")
            else:
                cls._task_context.task_id = previous_task_id

    @classmethod
    def current_task_id(cls) -> str | None:
        result = getattr(cls._task_context, "task_id", None)
        return result

    @classmethod
    def format_message(cls, message: str, start_timer: float | None) -> str:
        result = message
        if start_timer is not None:
            result = f"{message} ({time.perf_counter() - start_timer:.4f}s)"
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

    @classmethod
    def append_task_line(cls, lines: list[str], line: str) -> None:
        lines.append(line)
        while len(lines) > cls.max_task_lines:
            lines.pop(0)

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
            return

        rendered_message = cls.format_message(message, start_timer)
        level = cls._normalize_level(level_name)
        logging.getLogger(module_name).log(
            level,
            rendered_message,
            extra={"_shared_logger_managed": True},
        )

        active_task_id = task_id if task_id is not None else cls.current_task_id()
        if active_task_id is None or level < cls.frontend_level:
            return

        with cls._task_lock:
            lines = cls._task_buffers.get(active_task_id)
        if lines is None:
            return

        task_line = cls.format_task_line(
            module_name=module_name,
            level_name=level_name,
            message=rendered_message,
        )
        cls.append_task_line(lines, task_line)

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


def get_logger(module_name: str) -> ModuleLogger:
    result = SharedLogger.get_logger(module_name)
    return result


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
