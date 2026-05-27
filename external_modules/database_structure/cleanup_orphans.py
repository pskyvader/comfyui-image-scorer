"""Cleanup orphaned image / JSON companion files inside the scored folder.

Phase 1 – move EVERY file that lacks a companion in its own directory
         to the root ``scored/`` folder.
Phase 2 – at the root level only:
           * stems that now have both image + JSON → move the pair into
             ``scored_0.5/`` (standard initialiser tier).
           * stems that are still singletons → delete them if ``--delete``
             is active (default), otherwise leave them in place.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from shared.paths import image_root_processed  # noqa: E402
import time
logger = logging.getLogger(__name__)

logger: logging.Logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def _walk_all_files(Path) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            p = Path(dirpath) / name
            ext = p.suffix.lower()
            if ext in IMAGE_EXTENSIONS or ext == ".json":
                groups[p.stem].append(p)
    result = groups
    logger.debug("_walk_all_files took %.4fs", time.perf_counter() - _start)
    return result


def _scored_root_files(Path) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for f in root.iterdir():
        if not f.is_file():
            continue
        ext = f.suffix.lower()
        if ext in IMAGE_EXTENSIONS or ext == ".json":
            groups[f.stem].append(f)
    result = groups
    logger.debug("_scored_root_files took %.4fs", time.perf_counter() - _start)
    return result


def cleanup_orphans(
    _start = time.perf_counter()
    _start = time.perf_counter()
    root: Path | None = None,
    dry_run: bool = False,
    delete_enabled: bool = True,
    logger.debug("cleanup_orphans took %.4fs", time.perf_counter() - _start)
) -> int:
    if root is None:
        root = Path(image_root_processed)

    if not root.exists():
        logger.warning("Root does not exist: %s", root)
        return 0

    groups = _walk_all_files(root)

    orphans: list[Path] = []
    examples: list[str] = []

    # ── Phase 1 — collect orphans from the whole tree ──────────────────
    for stem, files in groups.items():
        images = [f for f in files if f.suffix.lower() in IMAGE_EXTENSIONS]
        jsons = [f for f in files if f.suffix.lower() == ".json"]

        if not images or not jsons:
            orphans.extend(files)
            continue

        dirs: dict[Path, list[Path]] = defaultdict(list)
        for f in files:
            dirs[f.parent].append(f)

        for d, d_files in dirs.items():
            has_img = any(f.suffix.lower() in IMAGE_EXTENSIONS for f in d_files)
            has_json = any(f.suffix.lower() == ".json" for f in d_files)
            if not (has_img and has_json):
                orphans.extend(d_files)

    # ── Phase 1b — move all orphans to root scored/ ────────────────────
    moved_to_root = 0
    with tqdm(
        total=len(orphans),
        desc="[CLEANUP] Moving orphans to root",
        unit="file",
        leave=False,
    ) as pbar:
        for f in orphans:
            if f.parent == root:
                pbar.update(1)
                continue
            dest = root / f.name
            if dest.exists():
                logger.warning("Target exists, skipping %s -> %s", f, dest)
                pbar.update(1)
                continue
            if not dry_run:
                os.replace(str(f), str(dest))
            moved_to_root += 1
            pbar.update(1)

    if moved_to_root and len(examples) < 3:
        examples.append(f"Moved {moved_to_root} orphan(s) to root {root.name}")

    # ── Phase 2 — process root-level files only ────────────────────────
    score_05 = root / "scored_0.5"
    root_groups = _scored_root_files(root)

    logger.debug("[CLEANUP] Phase 2: processing %d root-level stem(s)", len(root_groups))
    moved_to_05 = 0
    deleted = 0

    for stem, files in tqdm(
        root_groups.items(),
        desc="[CLEANUP] Resolving root orphans",
        unit="stem",
        leave=False,
    ):
        images = [f for f in files if f.suffix.lower() in IMAGE_EXTENSIONS]
        jsons = [f for f in files if f.suffix.lower() == ".json"]

        if images and jsons:
            if not dry_run:
                score_05.mkdir(parents=True, exist_ok=True)
            with tqdm(
                total=len(images) + len(jsons),
                desc=f"  Moving {stem} to {score_05.name}",
                unit="file",
                leave=False,
            ) as inner:
                for f in images + jsons:
                    dest = score_05 / f.name
                    if dest.exists():
                        inner.update(1)
                        continue
                    if not dry_run:
                        os.replace(str(f), str(dest))
                    moved_to_05 += 1
                    inner.update(1)
            if len(examples) < 3:
                examples.append(
                    f"{stem}: pair moved to {score_05.name}"
                )
        elif delete_enabled:
            if not dry_run:
                for f in files:
                    f.unlink(missing_ok=True)
            deleted += len(files)
            if len(examples) < 3:
                examples.append(f"{stem}: {len(files)} singleton(s) deleted")

    action = "Would " if dry_run else ""
    logger.info(
        "%sMoved %d orphan(s) to root, %s moved %d to %s, %sdeleted %d",
        action,
        moved_to_root,
        action,
        moved_to_05,
        score_05.name,
        action,
        deleted,
    )
    for ex in examples:
        logger.info("  e.g. %s", ex)

    return moved_to_root + moved_to_05 + deleted


def main() -> None:
    _start = time.perf_counter()
    _start = time.perf_counter()
    parser = argparse.ArgumentParser(
        description="Cleanup orphaned image/JSON companion files inside the scored folder"
    )
    parser.add_argument(
        "--root",
        default=None,
        help="Override root directory (default: config image_root_processed)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview only, no changes"
    )
    parser.add_argument(
        "--no-delete",
        action="store_true",
        help="Do NOT delete truly orphan singletons (default: delete them)",
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

    count = cleanup_orphans(
        root=Path(args.root) if args.root else None,
        dry_run=args.dry_run,
        delete_enabled=not args.no_delete,
    )

    if count:
        logger.info("Resolved %d file(s)", count)
    else:
        logger.info("No orphans to clean")
        logger.debug("main took %.4fs", time.perf_counter() - _start)


if __name__ == "__main__":
    main()
