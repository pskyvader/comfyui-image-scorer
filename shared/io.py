import json
import jsonlines
import os
from collections import deque
from pathlib import Path
from typing import (
    Any,
    Callable,
    Iterator,
    TypeVar,
)
from tqdm import tqdm
import time

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from .logger import get_logger, ModuleLogger

R = TypeVar("R")
logger: ModuleLogger = get_logger(__name__)


def load_single_jsonl(filename: str, skip_invalid: bool = True) -> Iterator[Any]:
    if not os.path.exists(filename):
        return

    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            stripped: str = line.strip()
            if not stripped:
                continue
            if skip_invalid:
                try:
                    yield json.loads(stripped)
                except json.JSONDecodeError:
                    continue
            else:
                yield json.loads(stripped)


def write_single_jsonl(filename: str, data: list[Any], mode: str) -> None:
    file_path = Path(filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if not mode.startswith("r"):
        with tqdm(total=len(data), delay=3.0) as pbar:
            with jsonlines.open(file_path, mode="w") as writer:
                for item in data:
                    writer.write(item)
                    pbar.update(1)


def parallel_batch(fn: Callable[..., R], items: list[tuple[Any, ...]]) -> list[R]:
    results: list[R] = []
    for item in items:
        results.append(fn(*item))
    return results


def parallel_for(
    fn: Callable[..., R],
    items: list[tuple[Any, ...]],
    *,
    max_workers: int = 1,
    batch_size: int = 0,
    desc: str = "Processing",
    unit: str = "items",
    on_progress: Callable[[], None] | None = None,
) -> list[R]:
    """Execute fn(*item) for each item across a thread pool.

    Args:
        fn: The callable to invoke for each item.
        items: Argument tuples, each unpacked as ``fn(*item)``.
        max_workers: Maximum number of concurrent threads.
        batch_size: If > 0, submit items in batches of this size.
        desc: tqdm description prefix.
        unit: tqdm unit label.
        on_progress: Optional callable invoked after each completed item.

    Returns:
        List of results in arbitrary (completion) order.
    """
    logger.info(f"starting parallel workers for {str(fn)[:10]}...")
    results: list[R] = []
    n: int = len(items)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(total=n, desc=desc, unit=unit, leave=False, position=0, delay=3.0) as pbar:
            try:
                if batch_size > 0:
                    batches = [
                        items[i : i + batch_size] for i in range(0, n, batch_size)
                    ]
                    futures_list: list[Future[list[R]]] = [
                        executor.submit(parallel_batch, fn, batch) for batch in batches
                    ]
                    for f in as_completed(futures_list):
                        res: list[R] = f.result()
                        results.extend(res)
                        pbar.update(len(res))
                        if on_progress:
                            on_progress()
                else:
                    started: dict[Future[R], float] = {
                        executor.submit(fn, *item): time.perf_counter()
                        for item in items
                    }
                    recent: deque[float] = deque(maxlen=100)
                    for future in as_completed(started):
                        elapsed = time.perf_counter() - started[future]
                        recent.append(elapsed)
                        results.append(future.result())
                        pbar.update(1)
                        pbar.set_postfix(avg=f"{sum(recent)/len(recent):.4f}s")
                        if on_progress:
                            on_progress()
            except KeyboardInterrupt:
                executor.shutdown(wait=False, cancel_futures=True)
                logger.warning("Interrupted after %d/%d %s", len(results), n, unit)
    return results


def discover_files(root: str) -> Iterator[tuple[str, str]]:
    _start = time.perf_counter()
    total_files = 0
    for dirpath, _, files in os.walk(root):
        file_set = set(files)
        total_files += len(files)
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                base = f.rsplit(".", 1)[0]
                json_name = base + ".json"

                if json_name in file_set:
                    yield (
                        os.path.join(dirpath, f),
                        os.path.join(dirpath, json_name),
                    )


def collect_single_file(
    file: tuple[str, str], processed_files: set[str], root: str
) -> tuple[str, dict[str, Any], str, str] | None:
    _start = time.perf_counter()
    img_path, meta_path = file

    file_id = os.path.basename(img_path)
    result: tuple[str, dict[str, Any], str, str] | None

    if file_id in processed_files or img_path in processed_files:
        result = None
    else:
        entry, err = load_json(meta_path, expect=dict)
        if err == "not_found" or entry is None:
            result = None
        elif err:
            result = None
        else:
            if "score" not in entry:
                keys = list(entry.keys())
                if keys:
                    first_key = keys[0]
                    if isinstance(entry[first_key], dict) and len(keys) < 5:
                        timestamp = first_key
                        entry = entry[first_key]
                    else:
                        timestamp = "unknown"
                else:
                    timestamp = "unknown"
            else:
                timestamp = next(iter(entry.keys())) if entry.keys() else "unknown"

            result = (img_path, entry, timestamp, file_id)

    return result


def collect_valid_files(
    files: Iterator[tuple[str, str]],
    processed_files: set[str],
    root: str,
    limit: int,
    max_workers: int,
    scored_only: bool,
) -> list[tuple[str, dict[str, Any], str, str]]:
    collected_data: list[tuple[str, dict[str, Any], str, str]] = []

    if files:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    collect_single_file,
                    file,
                    processed_files,
                    root,
                )
                for file in files
            ]
            # total=len(files)
            with tqdm(desc="Collecting", unit=" files", delay=3.0) as pbar:
                for future in as_completed(futures):
                    result = future.result()

                    pbar.update(1)
                    if result is None:
                        continue
                    if scored_only and ("score" not in result[1]):
                        continue

                    collected_data.append(result)

                    if limit > 0 and len(collected_data) >= limit:
                        for f in futures:
                            f.cancel()
                        executor.shutdown(wait=False, cancel_futures=True)
                        break

    return collected_data


def _recursive_parse_json(obj: Any, path: str | None) -> Any:
    _start = time.perf_counter()
    result: Any
    if isinstance(obj, dict):
        result = {k: _recursive_parse_json(v, path) for k, v in obj.items()}
    elif isinstance(obj, list):
        result = [_recursive_parse_json(v, path) for v in obj]
    elif isinstance(obj, str):
        s = obj.strip()
        if (s.startswith("{") and s.endswith("}")) or (
            s.startswith("[") and s.endswith("]")
        ):
            parsed = json.loads(obj)
            if isinstance(parsed, (dict, list)):
                result = _recursive_parse_json(parsed, path)
            else:
                result = parsed
        else:
            result = obj
    else:
        result = obj
    # if isinstance(result, (dict, list)):

    # else:

    return result


def load_json(
    path: str,
    expect: type | tuple[type, ...] | None,
) -> tuple[Any, str | None]:
    _start = time.perf_counter()
    result: tuple[Any, str | None]
    if not path:
        result = (None, "missing_path")
    elif not Path(path).exists():
        result = (None, "not_found")
    else:
        with Path(path).open("r", encoding="utf-8") as fh:
            data = json.load(fh)
            data = _recursive_parse_json(data, path)
        if expect is not None and not isinstance(data, expect):
            result = (None, "invalid_type")
        else:
            result = (data, None)

    return result


def atomic_write_json(path: str, data: Any, *, indent: int | None) -> None:
    _start = time.perf_counter()
    p: Path = Path(path)
    # logger.debug(f"saving: {path}")
    p.parent.mkdir(parents=True, exist_ok=True)

    tmp: Path = p.with_suffix(p.suffix + ".tmp")

    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=indent)

    os.replace(tmp, p)


def load_single_entry_mapping(
    path: str,
) -> tuple[dict[str, Any] | None, str | None, str | None]:
    _start = time.perf_counter()
    data, err = load_json(path, expect=dict)
    result: tuple[dict[str, Any] | None, str | None, str | None]
    if err:
        result = (None, None, err)
    elif data is None or len(data.keys()) != 1:
        result = (None, None, "invalid_keys")
    else:
        key = next(iter(data.keys()))
        payload = data[key]
        if not isinstance(payload, dict):
            result = (None, None, "invalid_payload")
        else:
            result = (payload, str(key), None)
    return result
