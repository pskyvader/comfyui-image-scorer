"""Image processor - handles discovery, conversion, and initialization of new images."""

import json
import os
import shutil
import sys
import time
import logging
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any

from tqdm import tqdm

# Add parent path for imports
_root = Path(__file__).parent.parent.parent  # comfyui-image-scorer
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


# Setup logging
logger = logging.getLogger(__name__)

# Import database functions
from shared.config import config
from database.images_table import (
    add_image,
    get_image as db_get_image,
    get_all_images,
    get_image_count,
    update_image_score_confidence,
    get_all_images as db_get_all_images,
    update_image_tags,
    update_image_confidence,
)
from database.comparisons_table import (
    add_historical_comparison,
    deduplicate_comparisons,
    get_comparison_count,
)
from shared.vectors.terms import extract_terms

# Use central paths definitions as the source of truth for filesystem roots
from shared.paths import (
    image_root,
    image_root_processed,
    output_dir,
)
from file_management.path_handler import compute_path_from_filename, get_ranked_root


class ImageProcessor:
    """Process uninitialized images with parallel workers."""

    def __init__(self, max_workers: int | None = None):
        # Load performance settings from config (strict no-default policy)
        ranking_conf = config.get("ranking")
        self.max_workers = max_workers or int(ranking_conf["max_workers"])
        self.batch_size = int(ranking_conf["batch_size"])
        self.default_score = float(ranking_conf["default_score"])
        self.default_confidence = float(ranking_conf["default_confidence"])
        self.reserve_count = int(ranking_conf["reserve_count"])

        self.processed_lock = Lock()
        self.processed_images = set()
        self.image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
        self.is_processing = False
        self.total_discovered = 0

        # Server-side LRU for recently shown images

        self.lru_size = int(ranking_conf["lru_size"])
        self.recent_images = deque(maxlen=self.lru_size)
        self.recent_chains = deque(maxlen=self.lru_size)
        self.recent_lock = Lock()

        # Load already processed filenames from database to avoid re-scanning
        self.sync_processed_images_from_db()

    def convert_old_score_to_new(
        self, old_score: float, modifier: float = 0.0
    ) -> float:
        """
        Convert old 1-5 score format to new 0-1 format.

        Old format: 1-5 integer with ±5 modifier
        New format: 0.0-1.0 float

        Conversion: (old_score - 1) / 4 + (modifier / 50)
        """
        # Normalize old 1-5 to 0-4
        normalized = old_score - 1.0

        # Apply modifier (convert from ±5 to ±0.1)
        modifier_adjusted = modifier / 50.0

        # Combine and clamp to 0-1
        new_score = (normalized / 4.0) + modifier_adjusted
        new_score = max(0.0, min(1.0, new_score))

        return round(new_score, 3)  # Keep 3 decimals precision

    def clean_json_metadata(
        self,
        json_data: dict[str, Any] | None,
        default_score: float | None = None,
        filename: str | None = None,
    ) -> dict[str, Any]:
        if default_score is None:
            default_score = self.default_score

        """
        Clean JSON metadata, removing ranking system fields and keeping only original data.

        Removes: score, score_modifier, volatility, image, comparison_count
        Keeps: custom_text, cfg, sampler, steps, lora, positive_prompt, negative_prompt, etc.
        Adds: confidence=0, comparison_count=0, history=[], prompt_tags=None
        """
        # Normalize various incoming JSON shapes into a simple top-level metadata
        # structure expected by other parts of the system. We support two common
        # shapes:
        #  - Per-timestamp entries: {"2026-...": { ... }}
        #  - Flat metadata: {"score":..., "confidence":..., "comparison_history": [...]})

        remove_fields = {
            "score",
            "score_modifier",
            "volatility",
            "image",
            "comparison_count",
        }

        # If no JSON was provided, create a minimal metadata blob
        if not isinstance(json_data, dict) or not json_data:
            base = {}
        else:
            # If top-level already contains ranking keys, assume it's already in
            # the flat format and use it as base.
            if (
                "comparison_history" in json_data
                or "confidence" in json_data
                or "comparison_count" in json_data
            ):
                base = {k: v for k, v in json_data.items() if k not in remove_fields}
            else:
                # Attempt to pick the first timestamp-like entry as the canonical
                # metadata source.
                base = {}
                for k, v in json_data.items():
                    if isinstance(v, dict):
                        for key, value in v.items():
                            if key not in remove_fields:
                                base[key] = value
                        break

        # Ensure essential ranking fields exist at top-level
        base.setdefault("score", round(default_score, 3))
        base.setdefault("confidence", self.default_confidence)
        base.setdefault("comparison_count", 0)
        base.setdefault("comparison_history", [])

        # Include filename when available
        if filename:
            base.setdefault("filename", filename)

        # Extract prompt tags
        prompt = base.get("positive_prompt")
        if prompt and isinstance(prompt, str):
            weighted_tags = extract_terms(prompt)
            # Filter weight >= 1.0 and format for DB
            tags = [tag for tag, weight in weighted_tags if weight >= 1.0]
            if tags:
                base["prompt_tags"] = ",".join(tags)

        return base

    def process_image_file(
        self, image_path: Path
    ) -> tuple[bool, str, float | None, str | None, bool, str | None]:
        """
        Process a single image file: convert score, clean JSON, move to tier folder.

        Returns: (success, message, new_score)
        """
        try:
            filename = image_path.name

            # The only valid JSON companion name is replacing the extension (e.g. imageA.png -> imageA.json)
            json_path = image_path.with_suffix(".json")

            if not json_path.exists():
                # Add to processed_images for this session so we don't get stuck in an infinite loop
                # on the same first 1000 broken files. User can restart server to retry.
                with self.processed_lock:
                    self.processed_images.add(filename)
                msg = f"Skipping {filename}: No JSON companion found at {json_path}"
                return False, msg, None, None, False, None

            # Check if image already processed
            if filename in self.processed_images:
                return False, f"Already processed", None, None, False, None

            # Default score from config
            new_score = self.default_score

            # Try to read JSON metadata
            json_data = None
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    json_data = json.load(f)

                # Detect old 1-5 scoring inside nested structures or flat entries
                if isinstance(json_data, dict):
                    # Flat score at top-level
                    top_score = json_data.get("score")
                    if isinstance(top_score, (int, float)):
                        if top_score > 1.0:
                            # Old 1-5 scale
                            modifier = json_data.get("score_modifier", 0)
                            new_score = self.convert_old_score_to_new(
                                top_score, modifier
                            )
                        else:
                            new_score = float(top_score)
                    else:
                        # Try nested timestamp entries
                        for timestamp_key, entry_data in json_data.items():
                            if isinstance(entry_data, dict):
                                old_score = entry_data.get("score", 3)
                                modifier = entry_data.get("score_modifier", 0)
                                new_score = self.convert_old_score_to_new(
                                    old_score, modifier
                                )
                                break

            except Exception as e:
                pass
                json_data = None

            if json_data is None:
                pass
            # Clean/normalize metadata into top-level shape and ensure ranking keys exist
            cleaned_json = self.clean_json_metadata(
                json_data, default_score=new_score, filename=filename
            )

            # If a database record already exists for this filename, prefer
            # the DB score and do not overwrite it. This preserves historical
            # scores when files are removed/re-added.
            db_entry = db_get_image(filename)

            if db_entry:
                # Use DB score as authoritative and sync JSON to it
                chosen_score = float(
                    db_entry.get("score", cleaned_json.get("score", new_score))
                )
                cleaned_json["score"] = round(chosen_score, 3)
            else:
                chosen_score = cleaned_json.get("score", new_score)

            # Compute destination path using path handler (handles nested subfolders)

            # Write cleaned JSON back to the source BEFORE moving files so metadata
            # is preserved even if the move fails. Use an atomic replace where
            # possible to avoid partial writes.
            # Ensure we only write if the file actually existed
            if json_path.exists():
                tmp_json = json_path.parent / (json_path.name + ".tmp")
                with open(tmp_json, "w", encoding="utf-8") as f:
                    json.dump(cleaned_json, f, indent=2, ensure_ascii=False)
                os.replace(str(tmp_json), str(json_path))

            # Compute destination path using authoritative chosen_score
            dest_image = compute_path_from_filename(filename, float(chosen_score))
            dest_dir = dest_image.parent
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_json = dest_image.with_suffix(".json")

            # Helper: safe move with retries
            def safe_move(
                src: Path, dst: Path, retries: int = 3, delay: float = 0.1
            ) -> bool:
                for attempt in range(1, retries + 1):
                    if not src.exists():
                        return dst.exists()
                    try:
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(src), str(dst))
                        return True
                    except FileNotFoundError:
                        if dst.exists():
                            return True
                        if attempt < retries:
                            time.sleep(delay)
                            continue
                        return False
                    except PermissionError:
                        if attempt < retries:
                            time.sleep(delay)
                            continue
                        return False
                    except Exception:
                        if attempt < retries:
                            time.sleep(delay)
                            continue
                        return False
                return False

            # If destination exists, check if identical by size (high-speed check)
            if dest_image.exists():
                src_size = image_path.stat().st_size
                dst_size = dest_image.stat().st_size
                if src_size == dst_size:
                    # Identical size: assume same file to avoid expensive MD5 hashing
                    try:
                        if json_path.exists():
                            moved_ok = safe_move(json_path, dest_json)
                            if moved_ok:
                                pass
                            else:
                                # If move failed but dest_json exists, treat as success
                                if dest_json.exists():
                                    pass
                                else:
                                    pass
                    except Exception as e:
                        pass
                    try:
                        if image_path.exists():
                            pass
                            image_path.unlink()

                    except Exception as e:
                        pass
                    # Mark as processed and record association with existing file
                    with self.processed_lock:
                        self.processed_images.add(dest_image.name)

                    return (
                        True,
                        f"Duplicate associated with existing file: {dest_image.name}",
                        float(chosen_score),
                        dest_image.name,
                        bool(db_entry),
                    )
                else:
                    # Different file with same name at destination: find unique name
                    stem = dest_image.stem
                    suffix = dest_image.suffix
                    i = 1
                    while True:
                        candidate = dest_dir / f"{stem}_{i}{suffix}"
                        if not candidate.exists():
                            dest_image = candidate
                            dest_json = dest_image.with_suffix(".json")
                            break
                        i += 1

            # Move image and JSON together. If a failure occurs we attempt a
            # best-effort rollback so the source is left in a consistent state.
            if not image_path.exists():

                return False, f"Source not found: {image_path}", None, None, False

            moved_image = False
            moved_json = False
            try:
                image_moved = safe_move(image_path, dest_image)
                if not image_moved:

                    return False, f"Image move failed", None, None, False
                moved_image = True

                if json_path.exists():
                    json_moved = safe_move(json_path, dest_json)
                    if not json_moved:
                        # rollback image
                        try:
                            if dest_image.exists() and not image_path.exists():
                                safe_move(dest_image, image_path)
                        except Exception as e2:
                            pass
                        return False, f"JSON move failed", None, None, False
                    moved_json = True

                # logger.info(f"Moved image+json: {filename} -> {dest_dir}")
            except Exception as e:

                # Attempt rollback
                try:
                    if moved_image and dest_image.exists() and not image_path.exists():
                        safe_move(dest_image, image_path)
                except Exception as e2:
                    pass
                try:
                    if moved_json and dest_json.exists() and not json_path.exists():
                        safe_move(dest_json, json_path)
                except Exception as e2:
                    pass
                return False, f"Move failed: {e}", None, None, False, None

            # Mark as processed and return
            with self.processed_lock:
                self.processed_images.add(dest_image.name)

            return (
                True,
                f"Processed successfully (score: {float(chosen_score):.3f})",
                float(chosen_score),
                dest_image.name,
                bool(db_entry),
                cleaned_json.get("prompt_tags"),
            )

        except Exception as e:
            return False, f"Error processing: {e}", None, None, False, None

    def sync_processed_images_from_db(self):
        """Load all filenames currently in the database into the processed set."""
        try:
            all_imgs = get_all_images()
            with self.processed_lock:
                self.processed_images.clear()
                for img in all_imgs:
                    self.processed_images.add(img["filename"])
            logger.info(
                f"Synchronized {len(self.processed_images)} processed images from database."
            )
        except Exception as e:
            pass
    def add_to_database(
        self,
        filename: str,
        score: float,
        confidence: float = 0.0,
        comparison_count: int = 0,
        prompt_tags: str | None = None,
    ) -> bool:
        """Add processed image to database."""
        try:
            add_image(
                filename,
                score=score,
                confidence=confidence,
                comparison_count=comparison_count,
                prompt_tags=prompt_tags,
            )
            return True
        except Exception as e:
            return False

    def get_fast_total_count(self, source_dir: str) -> int:
        """Fastest way to get total count of image files by skipping excluded roots."""
        source_path = Path(source_dir).resolve()
        count = 0

        # Get exclude roots to skip

        try:
            exclude_roots = {get_ranked_root().resolve(), Path(output_dir).resolve()}
        except Exception:
            exclude_roots = set()

        for root, dirs, files in os.walk(source_path):
            root_path = Path(root).resolve()

            # Prune directories we want to skip
            if root_path in exclude_roots:
                dirs[:] = []  # Clear dirs to prevent recursion into this branch
                continue

            for f in files:
                if any(f.lower().endswith(ext) for ext in self.image_extensions):
                    count += 1

        self.total_discovered = count
        return count

    def process_next_batch(
        self, source_dir: str | None = None, batch_size: int | None = None
    ) -> dict[str, Any]:
        """Processes the next batch of unprocessed images."""
        if self.is_processing:
            return {"status": "skipped", "message": "Already processing"}

        self.is_processing = True
        try:
            # Get current DB count for global progress reporting
            db_count = get_image_count()
            total_goal = getattr(self, "total_discovered", 0)

            if not source_dir:
                source_dir = image_root

            # Use provided batch_size or fall back to config
            if batch_size is None:
                batch_size = self.batch_size

            source_path = Path(source_dir).resolve()

            # Step 1: Fast count if not done yet
            if self.total_discovered == 0:

                self.get_fast_total_count(source_dir)


            # Step 2: Find the next batch of files not in memory
            # Get exclude roots
            exclude_roots: list[Path] = [
                Path(image_root_processed).resolve(),
                get_ranked_root().resolve(),
                Path(output_dir).resolve(),
            ]

            # Step 2: Find all files not in memory and pick the oldest ones
            candidates = []


            # Optimized search using os.walk with directory pruning
            for root, dirs, files in os.walk(source_path):
                root_path = Path(root).resolve()

                # PRUNE: skip ranked/output folders completely at the dir level
                if root_path in exclude_roots:
                    dirs[:] = []
                    continue

                for f in files:
                    if f in self.processed_images:
                        continue
                    if any(f.lower().endswith(ext) for ext in self.image_extensions):
                        candidates.append(root_path / f)

            if not candidates:

                return {"status": "complete", "added": 0}

            # Sort candidates by modification time (newest first) so we can preserve the most recent ones
            try:

                def get_mtime(p):
                    try:
                        return os.path.getmtime(p)
                    except OSError:
                        return 0

                candidates.sort(key=get_mtime, reverse=True)
            except Exception as e:
                pass


            # Reserve newest images to help ComfyUI keep its numbering sequence
            reserve_count = self.reserve_count
            if len(candidates) < reserve_count:
                pass
                return {"status": "complete", "added": 0, "message": "Images reserved"}

            # Take the oldest ones after skipping the newest images
            batch_files = candidates[reserve_count : reserve_count + batch_size]


            # Step 3: Process the batch
            stats = {"processed": 0, "added": 0, "errors": 0, "failed": []}

            # We use a combined description for tqdm to show progress
            # Global progress starts at current DB count
            current_global = db_count
            system_total = db_count + total_goal

            with tqdm(
                total=len(batch_files),
                desc="[SCANNER] Initializing...",
                unit="img",
                leave=False,
            ) as pbar:

                def update_desc():
                    pbar.set_description(
                        f"[SCANNER] Global: {current_global}/{system_total}"
                    )

                update_desc()

                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    future_to_file = {
                        executor.submit(self.process_image_file, img_path): img_path
                        for img_path in batch_files
                    }

                    for future in as_completed(future_to_file):
                        img_path = future_to_file[future]
                        filename = img_path.name

                        try:
                            (
                                success,
                                message,
                                score,
                                dest_name,
                                db_entry_exists,
                                prompt_tags,
                            ) = future.result()

                            if success:
                                with self.processed_lock:
                                    stats["processed"] += 1
                                    db_name = dest_name if dest_name else filename
                                    if score is not None and not db_entry_exists:
                                        if self.add_to_database(
                                            db_name, score, prompt_tags=prompt_tags
                                        ):
                                            stats["added"] += 1
                                            current_global += 1
                                            update_desc()
                                        else:
                                            stats["errors"] += 1
                            else:
                                with self.processed_lock:
                                    if "Already processed" not in message:
                                        if stats["errors"] < 5:
                                            pass
                                        elif stats["errors"] == 5:
                                            stats["failed"].append(f"{filename}: {message}")
                                        # Log the first few errors to avoid spam but give info
                                        if stats["errors"] < 5:
                                            pass



                            pbar.update(1)
                            pbar.set_postfix(file=filename[:15], added=stats["added"])
                        except Exception as e:

                            pbar.update(1)

            logger.info(
                f"[SCANNER] Batch complete. Added: {stats['added']}, Errors: {stats['errors']}"
            )
            return stats

        finally:
            self.is_processing = False

    def rebuild_database_from_ranked(self) -> dict[str, Any]:
        """
        Rebuild database from existing ranked images.

        Scans the ranked folder and adds all images to the database if not already present.
        Also syncs comparison history from JSON files to the database.
        This recovers from database loss by reading metadata from ranked image JSON files.

        Returns: Summary dict with statistics
        """

        ranked_root = get_ranked_root()
        if not ranked_root.exists():

            return {"recovered": 0, "skipped": 0, "errors": 0}

        # Step 0: Reorganize loose files into subfolders if precision folders exist
        self.reorganize_folder_structure()

        # Find all image files in ranked folder
        image_files = [
            p
            for p in ranked_root.rglob("*")
            if p.is_file() and p.suffix.lower() in self.image_extensions
        ]



        stats = {"recovered": 0, "skipped": 0, "synced_history": 0, "errors": 0}

        # Phase 1: Recover/Sync Images (Metadata only)
        with tqdm(
            total=len(image_files),
            desc="[DATABASE] Phase 1/2: Recovering Images",
            unit="img",
            leave=False,
        ) as pbar:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(
                        self._rebuild_single_image, img_path, phase=1
                    ): img_path
                    for img_path in image_files
                }
                for future in as_completed(futures):
                    res = future.result()
                    if res:
                        stats[res] = stats.get(res, 0) + 1
                    pbar.update(1)
                    pbar.set_postfix(rec=stats["recovered"], skip=stats["skipped"])

        # Phase 2: Sync Comparison History (Relationships)
        # Done after Phase 1 to ensure all images exist in DB to satisfy Foreign Key constraints
        with tqdm(
            total=len(image_files),
            desc="[DATABASE] Phase 2/2: Syncing History",
            unit="img",
            leave=False,
        ) as pbar:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(
                        self._rebuild_single_image, img_path, phase=2
                    ): img_path
                    for img_path in image_files
                }
                for future in as_completed(futures):
                    res = future.result()
                    if res == "synced_history":
                        stats["synced_history"] += 1
                    elif res == "errors":
                        stats["errors"] += 1
                    pbar.update(1)
                    pbar.set_postfix(sync=stats["synced_history"], err=stats["errors"])

        # Phase 3: Deduplicate comparisons (relaxed matching by pair+winner, not exact timestamp)

        try:
            dedup_count = deduplicate_comparisons()
            stats["deduplicated"] = dedup_count

        except Exception as e:
            pass
        # Phase 4: Fix comparison_count for ALL images from actual DB counts
        # This ensures counts are correct even if JSON was missing history or Phase 2 didn't run


        all_images = db_get_all_images()
        fixed_count = 0
        with tqdm(
            total=len(all_images),
            desc="[DATABASE] Fixing comparison counts",
            unit="img",
            leave=False,
        ) as pbar:
            for img in all_images:
                filename = img.get("filename")
                if not filename:
                    continue
                actual_count = get_comparison_count(filename)
                current_count = img.get("comparison_count", 0)
                if actual_count != current_count:
                    try:
                        update_image_confidence(
                            filename, img.get("confidence", 0.0), actual_count
                        )
                        fixed_count += 1
                    except Exception:
                        pass
                pbar.update(1)


        logger.info(
            f"Database rebuild complete: recovered={stats['recovered']}, "
            f"skipped={stats['skipped']}, synced_history={stats['synced_history']}, "
            f"errors={stats['errors']}"
        )
        # Sync processed images set so the scanner doesn't pick up recovered images
        self.sync_processed_images_from_db()
        return stats

    def _rebuild_single_image(self, image_path: Path, phase: int = 1) -> str | None:
        """Worker for rebuilding a single image entry. Returns stats key to increment."""

        filename = image_path.name
        try:
            # 1. Read metadata from JSON
            json_path = image_path.with_suffix(".json")
            if not json_path.exists():
                return "skipped" if phase == 1 else None

            metadata = {}
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
            except Exception:
                return "errors"

            score = float(metadata.get("score", self.default_score))
            confidence = float(metadata.get("confidence", self.default_confidence))
            comparison_count = int(metadata.get("comparison_count", 0))
            comparison_history = metadata.get("comparison_history", [])

            if (
                isinstance(comparison_history, list)
                and len(comparison_history) > comparison_count
            ):
                comparison_count = len(comparison_history)

            # Extract prompt tags
            prompt_tags = None
            prompt = metadata.get("positive_prompt")
            if prompt and isinstance(prompt, str):
                weighted_tags = extract_terms(prompt)
                # Filter weight >= 1.0 and format for DB
                # tags: list[str] = [tag for tag, weight in weighted_tags if weight >= 1.0]
                tags: list[str] = [tag for tag, _ in weighted_tags]

                if tags:
                    prompt_tags: str = ",".join(tags)

            if phase == 1:
                # PHASE 1: Handle images table
                existing_data = db_get_image(filename)
                if existing_data:
                    changed = False
                    # ONLY sync from JSON if JSON has more comparisons (meaning JSON is more recent/complete)
                    # This prevents old JSONs from overwriting a fresh database state.
                    if comparison_count > existing_data.get("comparison_count", 0):
                        try:
                            update_image_score_confidence(
                                filename, score, confidence, comparison_count
                            )
                            changed = True
                        except Exception:
                            pass

                    # Always try to sync prompt tags if they are missing or if we found them
                    if prompt_tags and existing_data.get("prompt_tags") != prompt_tags:
                        try:
                            update_image_tags(filename, prompt_tags)
                            changed = True
                        except Exception:
                            pass

                    return "synced_history" if changed else "skipped"
                else:
                    if self.add_to_database(
                        filename,
                        score,
                        confidence,
                        comparison_count,
                        prompt_tags=prompt_tags,
                    ):
                        return "recovered"
                    return "errors"

            elif phase == 2:
                # PHASE 2: Handle comparisons history
                if comparison_history:
                    for comp in comparison_history:
                        other = comp.get("other")
                        timestamp = comp.get("timestamp")
                        if not other or not timestamp:
                            continue
                        is_winner_flag = comp.get("winner")
                        winner_file = filename if is_winner_flag else other
                        try:
                            result = add_historical_comparison(
                                filename_a=filename,
                                filename_b=other,
                                winner=winner_file,
                                timestamp=timestamp,
                                weight=float(comp.get("weight", 1.0)),
                                transitive_depth=int(comp.get("transitive_depth", 0)),
                            )
                            if result == 0:
                                logger.debug(
                                    f"[REBUILD] Comparison not added: {filename} vs {other}"
                                )
                        except Exception as e:
                            logger.debug(
                                f"[REBUILD] Failed to add comparison {filename} vs {other}: {e}"
                            )

                    # Update comparison_count in images table to match actual comparisons in DB
                    actual_count = get_comparison_count(filename)
                    if actual_count > 0:
                        existing = db_get_image(filename)
                        if existing and actual_count != existing.get(
                            "comparison_count", 0
                        ):
                            try:
                                update_image_confidence(
                                    filename,
                                    existing.get("confidence", 0.0),
                                    actual_count,
                                )
                            except Exception:
                                pass

                    return "synced_history"
                return None

        except Exception:
            return "errors"

    def reorganize_folder_structure(self):
        """
        Cleanup the ranked folder by moving 'loose' files in tier folders
        into their respective precision subfolders if those subfolders exist.
        """

        ranked_root = get_ranked_root()
        if not ranked_root.exists():
            return

        logger.info("[SCANNER] Checking folder structure for loose files...")
        moved_count = 0

        # Iterate through base tier folders (scored_0.0 to scored_1.0)
        try:
            for tier_folder in ranked_root.glob("scored_*"):
                if not tier_folder.is_dir():
                    continue

                # Check if this folder has any high-precision subfolders
                # Use os.listdir for speed
                try:
                    items = os.listdir(tier_folder)
                except Exception:
                    continue

                has_subfolders = any(
                    item.startswith("scored_") and (tier_folder / item).is_dir()
                    for item in items
                )
                if not has_subfolders:
                    continue

                # This folder has subfolders! Any files at the root of this folder
                # should be moved into their correct subfolder.
                for item in items:
                    loose_file = tier_folder / item
                    if (
                        loose_file.is_file()
                        and loose_file.suffix.lower() in self.image_extensions
                    ):
                        # We need the score to know which subfolder to move it to
                        json_path = loose_file.with_suffix(".json")
                        score = self.default_score
                        if json_path.exists():
                            try:
                                with open(json_path, "r", encoding="utf-8") as f:
                                    meta = json.load(f)
                                    score = float(meta.get("score", self.default_score))
                            except Exception:
                                pass

                        # Compute correct target path (now with precision enforced)
                        target_path = compute_path_from_filename(loose_file.name, score)

                        if target_path != loose_file:
                            target_path.parent.mkdir(parents=True, exist_ok=True)
                            try:
                                shutil.move(str(loose_file), str(target_path))
                                if json_path.exists():
                                    shutil.move(
                                        str(json_path),
                                        str(target_path.with_suffix(".json")),
                                    )
                                moved_count += 1
                            except Exception as e:
                                logger.error(
                                    f"Failed to reorganize {loose_file.name}: {e}"
                                )
        except Exception as e:
            logger.error(f"Folder reorganization failed: {e}")

        if moved_count:
            logger.info(
                f"[SCANNER] Reorganized {moved_count} loose files into subfolders."
            )
