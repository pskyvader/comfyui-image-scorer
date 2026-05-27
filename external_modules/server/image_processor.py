"""Image processor - discovery, initialization, and rebuild flow for step01."""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any

from tqdm import tqdm

_root = Path(__file__).parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

logger = logging.getLogger(__name__)

from shared.config import config  # noqa: E402
from shared.paths import image_root, image_root_processed, output_dir  # noqa: E402
from shared.vectors.terms import extract_terms  # noqa: E402

from external_modules.database_structure.images_table import (  # noqa: E402
    add_image,
    get_all_images,
    get_image as db_get_image,
    get_image_count,
    reset_all_image_ratings,
    update_image_rating_state,
    update_image_tags,
)
from external_modules.database_structure.comparisons_table import (  # noqa: E402
    add_historical_comparison,
    get_all_comparisons,
    normalize_comparisons,
)
from external_modules.database_structure.path_handler import (  # noqa: E402
    compute_path_from_filename,
    get_ranked_root,
    sync_image_metadata_to_json,
)
from external_modules.database_structure.deduplicate_scored import (
    deduplicate_scored,
)  # noqa: E402
from external_modules.database_structure.cleanup_orphans import (
    cleanup_orphans,
)  # noqa: E402
from external_modules.comparison.algorithm.trueskill_rating import (  # noqa: E402
    INITIAL_MEAN,
    INITIAL_UNCERTAINTY,
    Rating,
    public_score_from_rating,
    update_ratings,
)


class ImageProcessor:
    """Process uninitialized images with parallel workers."""

def __init__(int | None = None):
        ranking_conf = config.get("ranking")
        self.max_workers = max_workers or int(ranking_conf["max_workers"])
        self.batch_size = int(ranking_conf["batch_size"])
        self.default_score = float(ranking_conf["default_score"])
        self.reserve_count = int(ranking_conf["reserve_count"])

        self.processed_lock = Lock()
        self.processed_images: set[str] = set()
        self.image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
        self.is_processing = False
        self.total_discovered = 0

        self.lru_size = int(ranking_conf["lru_size"])
        self.recent_images = deque(maxlen=self.lru_size)
        self.recent_chains = deque(maxlen=self.lru_size)
        self.recent_lock = Lock()
        self.sync_processed_images_from_db()

def _extract_prompt_tags(dict[str, Any]) -> str | None:
        prompt = data.get("positive_prompt")
        if not isinstance(prompt, str) or not prompt:
            result = None
            logger.debug("_extract_prompt_tags took %.4fs", time.perf_counter() - _start)
            return result
        result = prompt
        logger.debug("_extract_prompt_tags took %.4fs", time.perf_counter() - _start)
        return result
        # weighted_tags = extract_terms(prompt)
        # tags = [tag for tag, weight in weighted_tags if weight >= 1.0]
        # return ",".join(tags) if tags else None

    def clean_json_metadata(
        _start = time.perf_counter()
        _start = time.perf_counter()
        self,
        json_data: dict[str, Any] | None,
        default_score: float | None = None,
        filename: str | None = None,
        logger.debug("clean_json_metadata took %.4fs", time.perf_counter() - _start)
    ) -> dict[str, Any]:
        if default_score is None:
            default_score = self.default_score

        remove_fields = {
            "score",
            "score_modifier",
            "volatility",
            "confidence",
            "image",
            "comparison_count",
            "rating_mu",
            "rating_sigma",
        }

        if not isinstance(json_data, dict) or not json_data:
            base: dict[str, Any] = {}
        elif "comparison_history" in json_data or "comparison_count" in json_data:
            base = {k: v for k, v in json_data.items() if k not in remove_fields}
        else:
            base = {}
            for _, value in json_data.items():
                if isinstance(value, dict):
                    for key, item in value.items():
                        if key not in remove_fields:
                            base[key] = item
                    break

        base.setdefault("score", round(float(default_score), 3))
        base.setdefault("rating_mu", INITIAL_MEAN)
        base.setdefault("rating_sigma", INITIAL_UNCERTAINTY)
        base.setdefault("comparison_count", 0)
        base.setdefault("comparison_history", [])
        base.pop("confidence", None)
        base.pop("score_modifier", None)
        base.pop("volatility", None)

        if filename:
            base.setdefault("filename", filename)

        prompt_tags = self._extract_prompt_tags(base)
        if prompt_tags:
            base["prompt_tags"] = prompt_tags
        return base

    def process_image_file(
        _start = time.perf_counter()
        _start = time.perf_counter()
        self, image_path: Path
        logger.debug("process_image_file took %.4fs", time.perf_counter() - _start)
    ) -> tuple[bool, str, float | None, str | None, bool, str | None]:
        """Process a single raw image file into the ranked tree."""

        try:
            filename = image_path.name
            json_path = image_path.with_suffix(".json")

            if not json_path.exists():
                with self.processed_lock:
                    self.processed_images.add(filename)
                return (
                    False,
                    f"Skipping {filename}: missing JSON companion",
                    None,
                    None,
                    False,
                    None,
                )

            if filename in self.processed_images:
                return False, "Already processed", None, None, False, None

            try:
                with open(json_path, "r", encoding="utf-8") as handle:
                    json_data = json.load(handle)
            except Exception:
                json_data = None

            cleaned_json = self.clean_json_metadata(
                json_data, default_score=self.default_score, filename=filename
            )

            db_entry = db_get_image(filename)
            if db_entry:
                chosen_score = float(db_entry.get("score", self.default_score))
                cleaned_json["score"] = round(chosen_score, 3)
                cleaned_json["rating_mu"] = float(
                    db_entry.get("rating_mu", INITIAL_MEAN)
                )
                cleaned_json["rating_sigma"] = float(
                    db_entry.get("rating_sigma", INITIAL_UNCERTAINTY)
                )
                cleaned_json["comparison_count"] = int(
                    db_entry.get("comparison_count", 0)
                )
            else:
                chosen_score = self.default_score
                cleaned_json["score"] = round(chosen_score, 3)
                cleaned_json["rating_mu"] = INITIAL_MEAN
                cleaned_json["rating_sigma"] = INITIAL_UNCERTAINTY
                cleaned_json["comparison_count"] = 0

            tmp_json = json_path.parent / f"{json_path.name}.tmp"
            with open(tmp_json, "w", encoding="utf-8") as handle:
                json.dump(cleaned_json, handle, indent=2, ensure_ascii=False)
            os.replace(str(tmp_json), str(json_path))

            dest_image = compute_path_from_filename(filename, chosen_score)
            dest_image.parent.mkdir(parents=True, exist_ok=True)
            dest_json = dest_image.with_suffix(".json")

def safe_move(Path, dst: Path, retries: int = 3) -> bool:
                for attempt in range(retries):
                    if not src.exists():
                        result = dst.exists()
                        logger.debug("safe_move took %.4fs", time.perf_counter() - _start)
                        return result
                    try:
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(src), str(dst))
                        result = True
                        logger.debug("safe_move took %.4fs", time.perf_counter() - _start)
                        return result
                    except Exception:
                        if attempt == retries - 1:
                            result = False
                            logger.debug("safe_move took %.4fs", time.perf_counter() - _start)
                            return result
                        time.sleep(0.1 * (attempt + 1))
                result = False
                logger.debug("safe_move took %.4fs", time.perf_counter() - _start)
                return result

            if dest_image.exists() and image_path.exists():
                if image_path.stat().st_size == dest_image.stat().st_size:
                    if json_path.exists():
                        safe_move(json_path, dest_json)
                    try:
                        image_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                    with self.processed_lock:
                        self.processed_images.add(dest_image.name)
                    return (
                        True,
                        f"Duplicate associated with existing file: {dest_image.name}",
                        chosen_score,
                        dest_image.name,
                        bool(db_entry),
                        cleaned_json.get("prompt_tags"),
                    )

                stem = dest_image.stem
                suffix = dest_image.suffix
                index = 1
                while True:
                    candidate = dest_image.parent / f"{stem}_{index}{suffix}"
                    if not candidate.exists():
                        dest_image = candidate
                        dest_json = candidate.with_suffix(".json")
                        break
                    index += 1

            if not safe_move(image_path, dest_image):
                return False, "Image move failed", None, None, False, None
            if json_path.exists() and not safe_move(json_path, dest_json):
                return False, "JSON move failed", None, None, False, None

            with self.processed_lock:
                self.processed_images.add(dest_image.name)

            return (
                True,
                f"Processed successfully (score: {chosen_score:.3f})",
                chosen_score,
                dest_image.name,
                bool(db_entry),
                cleaned_json.get("prompt_tags"),
            )
        except Exception as exc:
            return False, f"Error processing: {exc}", None, None, False, None

    def sync_processed_images_from_db(self) -> None:
        _start = time.perf_counter()
        _start = time.perf_counter()
        try:
            all_imgs = get_all_images()
            with self.processed_lock:
                self.processed_images.clear()
                for img in all_imgs:
                    self.processed_images.add(img["filename"])
            logger.info(
                "Synchronized %s processed images from database.",
                len(self.processed_images),
            )
        except Exception:
            logger.debug("sync_processed_images_from_db took %.4fs", time.perf_counter() - _start)

    def add_to_database(
        _start = time.perf_counter()
        _start = time.perf_counter()
        self,
        filename: str,
        score: float,
        comparison_count: int = 0,
        prompt_tags: str | None = None,
        rating_mu: float = INITIAL_MEAN,
        rating_sigma: float = INITIAL_UNCERTAINTY,
        logger.debug("add_to_database took %.4fs", time.perf_counter() - _start)
    ) -> bool:
        try:
            return add_image(
                filename=filename,
                score=score,
                comparison_count=comparison_count,
                prompt_tags=prompt_tags,
                rating_mu=rating_mu,
                rating_sigma=rating_sigma,
            )
        except Exception:
            return False

def get_fast_total_count(str) -> int:
        source_path = Path(source_dir).resolve()
        count = 0
        try:
            exclude_roots = {get_ranked_root().resolve(), Path(output_dir).resolve()}
        except Exception:
            exclude_roots = set()

        for root, dirs, files in os.walk(source_path):
            root_path = Path(root).resolve()
            if root_path in exclude_roots:
                dirs[:] = []
                continue
            for file in files:
                if any(file.lower().endswith(ext) for ext in self.image_extensions):
                    count += 1
        self.total_discovered = count
        result = count
        logger.debug("get_fast_total_count took %.4fs", time.perf_counter() - _start)
        return result

    def process_next_batch(
        _start = time.perf_counter()
        _start = time.perf_counter()
        self, source_dir: str | None = None, batch_size: int | None = None
        logger.debug("process_next_batch took %.4fs", time.perf_counter() - _start)
    ) -> dict[str, Any]:
        if self.is_processing:
            return {"status": "skipped", "message": "Already processing"}

        self.is_processing = True
        try:
            db_count = get_image_count()
            total_goal = getattr(self, "total_discovered", 0)
            source_dir = source_dir or image_root
            batch_size = batch_size or self.batch_size

            if self.total_discovered == 0:
                self.get_fast_total_count(source_dir)

            source_path = Path(source_dir).resolve()
            exclude_roots = [
                Path(image_root_processed).resolve(),
                get_ranked_root().resolve(),
                Path(output_dir).resolve(),
            ]
            candidates: list[Path] = []
            for root, dirs, files in os.walk(source_path):
                root_path = Path(root).resolve()
                if root_path in exclude_roots:
                    dirs[:] = []
                    continue
                for file in files:
                    if file in self.processed_images:
                        continue
                    if any(file.lower().endswith(ext) for ext in self.image_extensions):
                        candidates.append(root_path / file)

            if not candidates:
                return {"status": "complete", "added": 0}

            candidates.sort(key=lambda path: os.path.getmtime(path), reverse=True)
            if len(candidates) < self.reserve_count:
                return {"status": "complete", "added": 0, "message": "Images reserved"}
            batch_files = candidates[
                self.reserve_count : self.reserve_count + batch_size
            ]

            stats = {"processed": 0, "added": 0, "errors": 0, "failed": []}
            current_global = db_count
            system_total = db_count + total_goal

            with tqdm(
                total=len(batch_files),
                desc="[SCANNER] Initializing...",
                unit="img",
                leave=False,
            ) as pbar:

                def update_desc() -> None:
                    _start = time.perf_counter()
                    _start = time.perf_counter()
                    pbar.set_description(
                        f"[SCANNER] Global: {current_global}/{system_total}"
                    )
                    logger.debug("update_desc took %.4fs", time.perf_counter() - _start)

                update_desc()
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    future_to_file = {
                        executor.submit(self.process_image_file, img_path): img_path
                        for img_path in batch_files
                    }
                    for future in as_completed(future_to_file):
                        filename = future_to_file[future].name
                        try:
                            (
                                success,
                                message,
                                score,
                                dest_name,
                                db_exists,
                                prompt_tags,
                            ) = future.result()
                            if success:
                                stats["processed"] += 1
                                db_name = dest_name or filename
                                if score is not None and not db_exists:
                                    if self.add_to_database(
                                        db_name, score, prompt_tags=prompt_tags
                                    ):
                                        stats["added"] += 1
                                        current_global += 1
                                        update_desc()
                                    else:
                                        stats["errors"] += 1
                            elif "Already processed" not in message:
                                stats["errors"] += 1
                                if len(stats["failed"]) < 5:
                                    stats["failed"].append(f"{filename}: {message}")
                        except Exception as exc:
                            stats["errors"] += 1
                            if len(stats["failed"]) < 5:
                                stats["failed"].append(f"{filename}: {exc}")
                        pbar.update(1)
                        pbar.set_postfix(file=filename[:15], added=stats["added"])
            return stats
        finally:
            self.is_processing = False

    def rebuild_database_from_ranked(self) -> dict[str, Any]:
        """Rebuild or repair the ranking database from ranked files and companion JSON."""

        ranked_root = get_ranked_root()
        if not ranked_root.exists():
            return {"recovered": 0, "skipped": 0, "history_imported": 0, "errors": 0}

        self.reorganize_folder_structure()

        ranked_root = get_ranked_root()
        logger.info("[DATABASE] Deduplicating scored images...")
        deduplicate_scored(root=ranked_root)
        logger.info("[DATABASE] Cleaning up orphan files...")
        cleanup_orphans(root=ranked_root)

        image_files = [
            path
            for path in ranked_root.rglob("*")
            if path.is_file() and path.suffix.lower() in self.image_extensions
        ]

        stats = {
            "recovered": 0,
            "skipped": 0,
            "history_imported": 0,
            "errors": 0,
            "missing_nodes_removed": 0,
            "self_links_removed": 0,
            "same_direction_duplicates_removed": 0,
            "contradictions_removed": 0,
            "ratings_recomputed": 0,
            "json_synced": 0,
        }

        with tqdm(
            total=len(image_files),
            desc="[DATABASE] Recovering images",
            unit="img",
            leave=False,
        ) as pbar:
            for image_path in image_files:
                result = self._recover_ranked_image(image_path)
                stats[result] = stats.get(result, 0) + 1
                pbar.update(1)

        valid_filenames = {img["filename"] for img in get_all_images()}

        with tqdm(
            total=len(image_files),
            desc="[DATABASE] Importing history",
            unit="img",
            leave=False,
        ) as pbar:
            for image_path in image_files:
                imported = self._import_history_from_json(image_path, valid_filenames)
                stats["history_imported"] += imported
                pbar.update(1)

        logger.debug("[DATABASE] Normalizing comparisons...")
        cleanup = normalize_comparisons()
        stats.update(cleanup)

        ratings_recomputed = self._recompute_ratings_from_database_history()
        stats["ratings_recomputed"] = ratings_recomputed

        all_comparisons = get_all_comparisons()
        all_images = get_all_images()

        filename_to_path: dict[str, Path] = {}
        for p in image_files:
            filename_to_path[p.name] = p

        filename_to_comparisons: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for comp in all_comparisons:
            filename_to_comparisons[comp["filename_a"]].append(comp)
            filename_to_comparisons[comp["filename_b"]].append(comp)

        filename_to_image_data: dict[str, dict[str, Any]] = {}
        for img in all_images:
            filename_to_image_data[img["filename"]] = img

        logger.debug(
            "[DATABASE] Syncing %d images (pre-built indexes ready)...",
            len(all_images),
        )
        with tqdm(
            total=len(all_images),
            desc="[DATABASE] Syncing JSON metadata",
            unit="img",
            leave=False,
        ) as pbar:
            for img in all_images:
                if sync_image_metadata_to_json(
                    filename=img["filename"],
                    score=float(img["score"]),
                    rating_mu=float(img.get("rating_mu", INITIAL_MEAN)),
                    rating_sigma=float(img.get("rating_sigma", INITIAL_UNCERTAINTY)),
                    comparison_count=int(img.get("comparison_count", 0)),
                    filename_to_path=filename_to_path,
                    filename_to_comparisons=filename_to_comparisons,
                    filename_to_image_data=filename_to_image_data,
                ):
                    stats["json_synced"] += 1
                pbar.update(1)

        self.sync_processed_images_from_db()
        logger.info("Database rebuild complete: %s", stats)
        return stats

        _start = time.perf_counter()
        _start = time.perf_counter()
    def _recover_ranked_image(self, image_path:
    logger.debug("rebuild_database_from_ranked took %.4fs", time.perf_counter() - _start)
        _start = time.perf_counter()
    logger.debug("rebuild_database_from_ranked took %.4fs", time.perf_counter() - _start)
        _start = time.perf_counter()
        result = Path) -> str:
        logger.debug("_recover_ranked_image took %.4fs", time.perf_counter() - _start)
        return result
        filename = image_path.name
        json_path = image_path.with_suffix(".json")
        metadata: dict[str, Any] = {}
        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as handle:
                    metadata = json.load(handle)
            except Exception:
                metadata = {}

        cleaned = self.clean_json_metadata(
            metadata, default_score=self.default_score, filename=filename
        )
        prompt_tags = cleaned.get("prompt_tags") or self._extract_prompt_tags(cleaned)
        existing = db_get_image(filename)
        if existing:
            if prompt_tags and existing.get("prompt_tags") != prompt_tags:
                update_image_tags(filename, prompt_tags)
            return "skipped"
        ok = self.add_to_database(
            filename=filename,
            score=self.default_score,
            comparison_count=0,
            prompt_tags=prompt_tags,
            rating_mu=INITIAL_MEAN,
            rating_sigma=INITIAL_UNCERTAINTY,
        )
        return "recovered" if ok else "errors"

    def _import_history_from_json(
        _start = time.perf_counter()
        _start = time.perf_counter()
        self, image_path: Path, valid_filenames: set[str] | None = None
        logger.debug("_import_history_from_json took %.4fs", time.perf_counter() - _start)
    ) -> int:
        json_path = image_path.with_suffix(".json")
        if not json_path.exists():
            return 0
        try:
            with open(json_path, "r", encoding="utf-8") as handle:
                metadata = json.load(handle)
        except Exception:
            return 0

        history = metadata.get("comparison_history", [])
        if not isinstance(history, list):
            return 0

        imported = 0
        filename = image_path.name
        if valid_filenames is not None and filename not in valid_filenames:
            return 0
        for comp in history:
            other = comp.get("other")
            timestamp = comp.get("timestamp")
            if not other or not timestamp:
                continue
            if valid_filenames is not None and other not in valid_filenames:
                continue
            winner_file = filename if comp.get("winner") else other
            if valid_filenames is not None and winner_file not in valid_filenames:
                continue
            result = add_historical_comparison(
                filename_a=filename,
                filename_b=other,
                winner=winner_file,
                timestamp=str(timestamp),
                weight=float(comp.get("weight", 1.0)),
                transitive_depth=int(comp.get("transitive_depth", 0)),
            )
            if result:
                imported += 1
        return imported

    def _recompute_ratings_from_database_history(self) -> int:
        _start = time.perf_counter()
        _start = time.perf_counter()
        reset_all_image_ratings(score=self.default_score)
        all_images = get_all_images()
        ratings = {
            img["filename"]: Rating(mu=INITIAL_MEAN, sigma=INITIAL_UNCERTAINTY)
            for img in all_images
        }
        counts = {img["filename"]: 0 for img in all_images}
        last_seen: dict[str, str] = {}

        all_comparisons = get_all_comparisons()
        with tqdm(
            total=len(all_comparisons),
            desc="[DATABASE] Recomputing ratings",
            unit="cmp",
            leave=False,
        ) as pbar:
            for comp in all_comparisons:
                left = comp["filename_a"]
                right = comp["filename_b"]
                winner = comp["winner"]
                if (
                    left not in ratings
                    or right not in ratings
                    or winner not in (left, right)
                ):
                    pbar.update(1)
                    continue
                loser = right if winner == left else left
                winner_rating, loser_rating = update_ratings(
                    ratings[winner], ratings[loser]
                )
                ratings[winner] = winner_rating
                ratings[loser] = loser_rating
                counts[winner] += 1
                counts[loser] += 1
                timestamp = str(comp.get("timestamp", ""))
                last_seen[winner] = timestamp
                last_seen[loser] = timestamp
                pbar.update(1)

        updated = 0
        with tqdm(
            total=len(ratings),
            desc="[DATABASE] Updating scores",
            unit="img",
            leave=False,
        ) as pbar:
            for filename, rating in ratings.items():
                if update_image_rating_state(
                    filename=filename,
                    score=public_score_from_rating(rating),
                    rating_mu=rating.mu,
                    rating_sigma=rating.sigma,
                    comparison_count=counts.get(filename, 0),
                    touch_timestamp=False,
                    last_compared_at=last_seen.get(filename),
                ):
                    updated += 1
                pbar.update(1)
        logger.debug("[DATABASE] Recomputed ratings for %d images.", updated)
        result = updated
        logger.debug("_recompute_ratings_from_database_history took %.4fs", time.perf_counter() - _start)
        result = result
        logger.debug("_recompute_ratings_from_database_history took %.4fs", time.perf_counter() - _start)
        return result

    def reorganize_folder_structure(self) -> None:
        _start = time.perf_counter()
        _start = time.perf_counter()
        ranked_root = get_ranked_root()
        if not ranked_root.exists():
            logger.debug("reorganize_folder_structure took %.4fs", time.perf_counter() - _start)
            logger.debug("reorganize_folder_structure took %.4fs", time.perf_counter() - _start)
            return

        logger.info("[SCANNER] Checking folder structure for loose files...")

        moves: list[tuple[Path, Path]] = []
        try:
            for tier_folder in ranked_root.glob("scored_*"):
                if not tier_folder.is_dir():
                    continue
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
                for item in items:
                    loose_file = tier_folder / item
                    if (
                        not loose_file.is_file()
                        or loose_file.suffix.lower() not in self.image_extensions
                    ):
                        continue
                    json_path = loose_file.with_suffix(".json")
                    score = self.default_score
                    if json_path.exists():
                        try:
                            with open(json_path, "r", encoding="utf-8") as handle:
                                meta = json.load(handle)
                            score = float(meta.get("score", self.default_score))
                        except Exception:
                            pass
                    target_path = compute_path_from_filename(loose_file.name, score)
                    if target_path == loose_file:
                        continue
                    moves.append((loose_file, target_path))
        except Exception as exc:
            logger.error("Folder reorganization scan failed: %s", exc)
            logger.debug("reorganize_folder_structure took %.4fs", time.perf_counter() - _start)
            logger.debug("reorganize_folder_structure took %.4fs", time.perf_counter() - _start)
            return

        moved_count = 0
        with tqdm(
            total=len(moves),
            desc="[SCANNER] Reorganizing files",
            unit="file",
            leave=False,
        ) as pbar:
            for loose_file, target_path in moves:
                try:
                    json_path = loose_file.with_suffix(".json")
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(loose_file), str(target_path))
                    if json_path.exists():
                        shutil.move(
                            str(json_path), str(target_path.with_suffix(".json"))
                        )
                    moved_count += 1
                except Exception as exc:
                    logger.warning("Failed to move %s: %s", loose_file.name, exc)
                pbar.update(1)

        if moved_count:
            logger.info(
                "[SCANNER] Reorganized %s loose files into subfolders.", moved_count
            )
