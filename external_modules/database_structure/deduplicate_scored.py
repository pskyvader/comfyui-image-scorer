"""Deduplicate scored images by comparing companion image MD5 across tier folders.

Resolves cases where the same or different images share a filename across
scored_* folders. For identical images, comparison_history is consolidated
into the newest copy and older copies are removed. For different images,
the newer copy is renamed with a _2 suffix.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

# sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from ...shared.io import (
    atomic_write_json,
    discover_files,
    load_json,
    parallel_for,
)  # noqa: E402
import time
from ...shared.logger import get_logger, ModuleLogger
from ...shared.config import config

logger: ModuleLogger = get_logger(__name__)
JsonDict = dict[str, Any]
EntryTriple = tuple[Path, Path, JsonDict]
_EXAMPLE_COUNT = 3


def _md5(path: Path) -> str:
    result = hashlib.md5(path.read_bytes()).hexdigest()
    return result


def _merge_comparison_histories(
    keeper: JsonDict, discard: list[JsonDict], filename: str
) -> None:
    seen_ids: set[int] = set()
    merged: list[JsonDict] = []

    for entry in keeper.get("comparison_history", []):
        cid: Any = entry.get("comparison_id")
        if cid is not None:
            seen_ids.add(cid)
        merged.append(entry)

    added = 0
    for old in discard:
        for entry in old.get("comparison_history", []):
            cid = entry.get("comparison_id")
            if cid is not None and cid not in seen_ids:
                seen_ids.add(cid)
                merged.append(entry)
                added += 1

    merged.sort(
        key=lambda e: (e.get("timestamp", "") or "", e.get("comparison_id") or 0)
    )
    keeper["comparison_history"] = merged
    keeper["comparison_count"] = len(merged)
    if added:
        logger.info(
            "Merged %d new comparison(s) from older copies of %s", added, filename
        )


def deduplicate_scored(
    root: Path | None = None,
    image_root_processed: Path | None = None,
    dry_run: bool = False,
    limit: int = 0,
) -> int:
    if root is None:
        root = Path(image_root_processed)

    if not root.exists():
        logger.warning("Scored root does not exist: %s", root)
        result = 0
        return result
    logger.debug("deduplicating entries...")
    file_pairs: list[tuple[str, str]] = list(discover_files(str(root)))

    def _scan_worker(
        image_path: str, json_path: str
    ) -> tuple[str, Path, Path, JsonDict] | None:
        data, err = load_json(json_path, expect=dict, default=None)
        if err is not None or data is None:
            return None
        img = Path(image_path)
        return (img.stem, img, Path(json_path), data)

    prepare_conf = config["prepare"]
    results = parallel_for(
        _scan_worker,
        file_pairs,
        max_workers=int(prepare_conf["max_workers"]),
        batch_size=int(prepare_conf["batch_size"]),
        desc="Scanning for duplicates",
        unit="files",
    )

    groups: dict[str, list[EntryTriple]] = defaultdict(list)
    for r in results:
        if r is not None:
            stem, img, jf, data = r
            groups[stem].append((img, jf, data))

    duplicates: dict[str, list[EntryTriple]] = {
        k: v for k, v in groups.items() if len(v) > 1
    }
    if not duplicates:
        logger.info("No duplicate filenames found under %s", root)
        result = 0
        return result

    logger.info("Found %d duplicate basename(s) to resolve", len(duplicates))

    if limit and limit < len(duplicates):
        logger.info("Processing first %d of %d group(s)", limit, len(duplicates))

    total_removed = 0
    total_renamed = 0
    examples: list[str] = []

    items_iter = (
        sorted(duplicates.items())[:limit] if limit else sorted(duplicates.items())
    )
    for basename, items in tqdm(items_iter, desc="Resolving duplicates", unit="group"):
        md5_groups: dict[str, list[EntryTriple]] = defaultdict(list)
        for img, jf, data in items:
            key = _md5(img)
            md5_groups[key].append((img, jf, data))

        for same_images in md5_groups.values():
            if len(same_images) < 2:
                continue

            same_images.sort(key=lambda x: x[0].stat().st_mtime, reverse=True)
            _keeper_img, keeper_jf, keeper_data = same_images[0]
            discard_list = same_images[1:]

            if len(examples) < _EXAMPLE_COUNT:
                examples.append(
                    f"{basename}: {len(same_images)} copy(ies), keeping {keeper_jf.parent.name}"
                )

            if not dry_run:
                for _img, _jf, ddata in discard_list:
                    _merge_comparison_histories(keeper_data, [ddata], basename)

                atomic_write_json(str(keeper_jf), keeper_data, indent=2)

                for dimg, djf, _ddata in discard_list:
                    dimg.unlink(missing_ok=True)
                    djf.unlink(missing_ok=True)

            total_removed += len(discard_list)

        if len(md5_groups) > 1:
            md5_keys: list[str] = list(md5_groups.keys())
            all_items: list[tuple[Path, Path, str]] = []
            for mk, mil in md5_groups.items():
                for img, jf, _data in mil:
                    all_items.append((img, jf, mk))

            all_items.sort(key=lambda x: x[0].stat().st_mtime)

            for img, jf, mk in all_items[1:]:
                if mk == md5_keys[0]:
                    continue

                stem = jf.stem
                if stem.endswith("_2"):
                    continue

                new_stem = f"{stem}_2"
                new_jf = jf.with_stem(new_stem)
                new_img = img.with_stem(new_stem)

                if new_jf.exists() or new_img.exists():
                    logger.warning(
                        "Cannot rename %s -> %s, target already exists",
                        jf.name,
                        new_jf.name,
                    )
                    continue

                if not dry_run:
                    os.replace(str(jf), str(new_jf))
                    os.replace(str(img), str(new_img))

                total_renamed += 1

    action = "Would remove" if dry_run else "Removed"
    logger.info(
        "%s %d duplicate(s), renamed %d conflict(s)",
        action,
        total_removed,
        total_renamed,
    )
    for ex in examples:
        logger.info("  e.g. %s", ex)

    result = total_removed + total_renamed
    return result


def main() -> None:
    _start = time.perf_counter()
    _start = time.perf_counter()
    parser = argparse.ArgumentParser(
        description="Deduplicate scored images by MD5 comparison"
    )
    parser.add_argument(
        "--root",
        default=None,
        help="Override scored root directory (default: config image_root_processed)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview only, no changes"
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Process only first N duplicate groups"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    count = deduplicate_scored(
        root=Path(args.root) if args.root else None,
        dry_run=args.dry_run,
        limit=args.limit,
    )

    if count:
        logger.info("Resolved %d duplicate file(s)", count)
    else:
        logger.info("No duplicates to resolve")


if __name__ == "__main__":
    main()
