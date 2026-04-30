import json
import jsonlines
import os
from pathlib import Path
from typing import (
    Any,
    TypeVar,
    Iterator,
)
from tqdm import tqdm
import ast

from concurrent.futures import ThreadPoolExecutor, as_completed


def load_single_jsonl(filename: str) -> list[Any]:
    data: list[Any] = []
    if os.path.exists(filename):
        with jsonlines.open(filename, mode="r") as reader:
            for obj in reader:
                data.append(obj)
    return data


def write_single_jsonl(filename: str, data: list[Any], mode: str) -> None:
    file_path = Path(filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if mode.startswith("r"):
        return  # Read-only mode, nothing to write
    
    with jsonlines.open(file_path, mode="w") as writer:
        for item in data:
            writer.write(item)


def discover_files(root: str) -> Iterator[tuple[str, str]]:
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
    file: tuple[str, str], processed_files: set[str], root: str
) -> tuple[str, dict[str, Any], str, str] | None:
    img_path, meta_path = file

    file_id = os.path.basename(img_path)
    if file_id in processed_files or img_path in processed_files:
        return None

    entry, err = load_json(meta_path, expect=dict, default=None)
    if err == "not_found" or entry is None:
        return None
    if err:
        print(f"Warning: Error while loading {file_id}: {err}")
        return None

    # Only drill down if 'score' is not at the root and there's a likely timestamp key
    if "score" not in entry:
        keys = list(entry.keys())
        if keys:
            first_key = keys[0]
            # Heuristic: if first key is a dict and root has few keys, it might be the old timestamp format
            if isinstance(entry.get(first_key), dict) and len(keys) < 5:
                timestamp = first_key
                entry = entry[first_key]
            else:
                timestamp = "unknown"
        else:
            timestamp = "unknown"
    else:
        # If 'score' is at root, use the first key as timestamp if it looks like one, or default
        timestamp = next(iter(entry.keys())) if entry.keys() else "unknown"

    # temppral fox
    if "comparison_count" in entry and entry["comparison_count"] > 5:
        if not "volatility" in entry:
            entry["volatility"] = 0
            entry["comparison_count"] = 5
    # temporal fix

    return (img_path, entry, timestamp, file_id)


def collect_valid_files(
    files: list[tuple[str, str]],
    processed_files: set[str],
    root: str,
    limit: int = 0,
    max_workers: int | None = None,
    scored_only: bool = True,
) -> list[tuple[str, dict[str, Any], str, str]]:

    collected_data: list[tuple[str, dict[str, Any], str, str]] = []

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
                    if scored_only and ("score" not in result[1]):
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


def _recursive_parse_json(obj: Any, path: str | None = None) -> Any:
    if isinstance(obj, dict):
        return {k: _recursive_parse_json(v, path) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_recursive_parse_json(v, path) for v in obj]
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
                    raise Warning(
                        f"parse python dictionary failed for {path} with data: {obj} . {e}"
                    )

            if isinstance(parsed, (dict, list)):
                return _recursive_parse_json(parsed, path)

    return obj


def load_json(
    path: str,
    expect: type | tuple[type, ...] | None,
    default: T | None,
) -> tuple[T | Any, str | None]:
    if not path:
        return default, "missing_path"
    path_obj = Path(path)
    if not path_obj.exists():
        return default, "not_found"

    try:
        with path_obj.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
            data = _recursive_parse_json(data, str(path_obj))
    except Exception as e:
        print(f"error parsing json file:{path_obj}, {e}")
        return default, "parse_error"

    if expect is not None and not isinstance(data, expect):
        return default, "invalid_type"
    return data, None


def atomic_write_json(path: str, data: Any, *, indent: int | None = None) -> None:
    p: Path = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    tmp: Path = p.with_suffix(p.suffix + ".tmp")

    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=indent)

    os.replace(tmp, p)


def load_single_entry_mapping(
    path: str,
) -> tuple[dict[str, Any] | None, str | None, str | None]:
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
