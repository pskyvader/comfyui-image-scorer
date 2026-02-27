import json
import jsonlines
import os
from pathlib import Path
from typing import (
    Any,
    Tuple,
    TypeVar,
    Iterator,
    List,
    Set,
    Dict,
    Optional,
)
from tqdm import tqdm
import ast

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict, Any, Optional, Set
from tqdm import tqdm
import os


def load_single_jsonl(filename: str) -> List[Any]:
    data: List[Any] = []
    if os.path.exists(filename):
        with jsonlines.open(filename, mode="r") as reader:
            for obj in reader:
                data.append(obj)
    return data


def write_single_jsonl(filename: str, data: List[Any], mode: str) -> None:
    file_path = Path(filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with jsonlines.open(file_path, mode=mode) as writer:
        writer.write_all(data)


def discover_files(root: str) -> Iterator[Tuple[str, str]]:
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
    print(f"{total_files} discovered files", flush=True)


def collect_single_file(
    file: Tuple[str, str], processed_files: Set[str], root: str
) -> Optional[Tuple[str, Dict[str, Any], str, str]]:
    img_path, meta_path = file

    file_id = os.path.relpath(img_path, root).replace("\\", "/")
    if file_id in processed_files:
        return None
    entry, err = load_json(meta_path, expect=dict, default=None)
    if err:
        raise FileNotFoundError(f"Error while loading {file_id}: {err}")
    if entry is None:
        raise FileNotFoundError(f"File {file_id} not found or invalid")

    timestamp = next(iter(entry.keys()))

    entry = entry[timestamp] if timestamp in entry else entry

    return (img_path, entry, timestamp, file_id)


def collect_valid_files(
    files: List[Tuple[str, str]],
    processed_files: Set[str],
    root: str,
    limit: int = 0,
    max_workers: Optional[int] = None,
    scored_only: bool = False,
) -> List[Tuple[str, Dict[str, Any], str, str]]:

    collected_data: List[Tuple[str, Dict[str, Any], str, str]] = []

    if not files:
        return collected_data

    # Good default for I/O-bound workloads
    if max_workers is None:
        cpu = os.cpu_count() or 1
        max_workers = min(32, cpu * 5)

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

        try:
            with tqdm(total=len(files), desc="Collecting", unit=" files") as pbar:
                for future in as_completed(futures):
                    result = future.result()  # raises immediately on error

                    pbar.update(1)
                    if result is None:
                        continue
                    if scored_only and "score" not in result:
                        continue

                    collected_data.append(result)

                    if limit > 0 and len(collected_data) >= limit:
                        for f in futures:
                            f.cancel()
                        executor.shutdown(wait=False, cancel_futures=True)

                        break

        except Exception:
            for f in futures:
                f.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            raise
    print(f"{len(collected_data)} Collected files", flush=True)

    return collected_data


T = TypeVar("T")


def _recursive_parse_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _recursive_parse_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_recursive_parse_json(v) for v in obj]
    elif isinstance(obj, str):
        s = obj.strip()
        if (s.startswith("{") and s.endswith("}")) or (
            s.startswith("[") and s.endswith("]")
        ):
            try:
                parsed = json.loads(obj)
            except Exception:
                # print(
                #     f"parse json failed for {obj} . {e}. Retrying as python dictionary"
                # )
                try:
                    parsed = ast.literal_eval(obj)
                except Exception as e:
                    raise Warning(f"parse python dictionary failed for {obj} . {e}")

            if isinstance(parsed, (dict, list)):
                return _recursive_parse_json(parsed)

    return obj


def load_json(
    path: str,
    expect: type | tuple[type, ...] | None,
    default: T | None,
) -> Tuple[T | Any, str | None]:
    if not path:
        return default, "missing_path"
    path_obj = Path(path)
    if not path_obj.exists():
        return default, "not_found"

    with path_obj.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
        data = _recursive_parse_json(data)

    if expect is not None and not isinstance(data, expect):
        return default, "invalid_type"
    return data, None


def atomic_write_json(path: str, data: Any, *, indent: int | None) -> None:
    print(f"Writing data to {path}...")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp_path = f"{path}.tmp"
    print(f"Using temporary file {tmp_path} for atomic write...")
    try:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=indent)
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except Exception:
                pass
        print(f"Replacing {tmp_path} with {path}...")
        os.replace(tmp_path, path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def load_single_entry_mapping(
    path: str,
) -> Tuple[dict[str, Any] | None, str | None, str | None]:
    data, err = load_json(path, expect=dict, default=None)
    if err:
        return None, None, err
    if data is None or len(data.keys()) != 1:
        return None, None, "invalid_keys"
    key = next(iter(data.keys()))
    payload = data[key]
    if not isinstance(payload, dict):
        return None, None, "invalid_payload"
    return (payload, str(key), None)
