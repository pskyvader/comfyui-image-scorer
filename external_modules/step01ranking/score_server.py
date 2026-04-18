import sys
import os
from pathlib import Path
import random
from threading import Thread
from time import sleep
from flask import Flask, request, send_from_directory, jsonify, send_file
from typing import Any
import argparse

if __name__ == "__main__":
    root_path = str(Path(__file__).parents[2])
    sys.path.insert(0, root_path)
    if __package__ is None:
        __package__ = "external_modules.step01ranking"


from .utils import (
    get_unscored_images,
    serve_file,
    image_root,
    scan_batch,
)

from .scores import submit_scores_handler
from .comparison import get_paired_images, apply_comparison_and_write
from .cache import (
    get_absolute_total,
    total_cached,
)

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent

# ───────────────────────────────
# Background scanning worker
# ───────────────────────────────
_scan_thread = None


def background_scan(batch_size: int = 1000):
    """
    Continuously scan images in batches until all are processed.
    Updates the cache and ensures valid/unscored counts are accurate.
    """
    retries = 0

    root = image_root()
    cached_total = 0
    total = 0
    while total == 0 or cached_total < total:
        # print(f"retries: {retries}, stop scan:{_stop_scan}")
        added = scan_batch(root, limit=batch_size)
        if not added:
            if retries > 3:
                print("stop scan")
                break
            sleep(0.1)  # wait a bit if no new images
            valid_images = get_unscored_images(root)
            if len(valid_images) > 0:
                retries += 1

            cached_total = total_cached()  # All images in cache
            total = get_absolute_total()

        else:
            retries = 0
            sleep(0.5)  # small delay for incremental updates


def trigger_scan():
    """Start the background scan if not running."""
    global _scan_thread
    if _scan_thread is None or not _scan_thread.is_alive():
        _scan_thread = Thread(target=background_scan, daemon=True)
        _scan_thread.start()


# ───────────────────────────────
# Static file routes
# ───────────────────────────────
@app.route("/")
def serve_index():
    return send_from_directory(str(BASE_DIR) + "/frontend/html", "index.html")


# Updated routes for your folder structure
@app.route("/css/<path:filename>")
def serve_css(filename: str):
    return send_from_directory(os.path.join(BASE_DIR, "frontend", "css"), filename)


@app.route("/js/<path:filename>")
def serve_js(filename: str):
    return send_from_directory(os.path.join(BASE_DIR, "frontend", "js"), filename)


# This route handles ALL your html files automatically
@app.route("/<path:filename>")
def serve_pages(filename: str):
    return send_from_directory(os.path.join(BASE_DIR, "frontend", "html"), filename)


@app.route("/image/<path:subpath>")
def serve_image_route(subpath: str):
    return serve_file(subpath)


@app.route("/thumbnail/<path:subpath>")
def serve_thumbnail_route(subpath: str):
    """
    Serve thumbnail for image. Falls back to full image if thumbnail generation fails.
    Generates thumbnails on-the-fly and caches them.
    """
    import io
    from PIL import Image

    try:
        root = image_root()
        # Build full path
        full_path = os.path.join(root, subpath.replace("/", os.sep))

        if not os.path.exists(full_path):
            return "Not found", 404

        # Try to generate thumbnail
        try:
            img = Image.open(full_path)
            img.thumbnail((200, 200), Image.Resampling.LANCZOS)

            # Serve thumbnail in memory
            img_io = io.BytesIO()
            img.save(img_io, format="JPEG", quality=85)
            img_io.seek(0)

            return send_file(img_io, mimetype="image/jpeg")
        except Exception as thumb_error:
            # Fall back to full image if thumbnail fails
            print(f"Thumbnail generation failed for {subpath}: {thumb_error}")
            return serve_file(subpath)
    except Exception as e:
        print(f"Error serving thumbnail for {subpath}: {e}")
        return "Error", 500


@app.route("/metadata/<path:subpath>")
def serve_metadata_route(subpath: str):
    return serve_file(subpath)


# ───────────────────────────────
# Status endpoint
# ───────────────────────────────
@app.route("/status")
def status():
    from .cache import get_comparison_stats, total_cached_unscored

    trigger_scan()

    unscored = total_cached_unscored()  # Images with no score
    comp_stats = get_comparison_stats()

    # scored_uncompared = images with score but comparison_count < 10
    uncompared = comp_stats.get("not_compared", 0)

    # compared = images with comparison_count >= 10
    compared = comp_stats.get("fully_compared", 0)
    partially_compared = comp_stats.get("partially_compared", 0)

    cached = total_cached()  # All images in cache
    total = get_absolute_total()  # Total images on disk
    # scanning = is_scanning()

    return jsonify(
        {
            "unscored": unscored,
            "uncompared": uncompared,
            "partially_compared": partially_compared,
            "compared": compared,
            "cached": cached,
            "total": total,
            # "scanning": scanning,
        }
    )


# ───────────────────────────────
# Gallery API endpoint
# ───────────────────────────────
@app.route("/api/scores")
def get_scores():
    """
    Get all scored images with their metadata for gallery view.
    Returns JSON array of scored images with score, modifier, comparison count, volatility.

    Query Parameters:
        page: Page number (1-based), default 1
        per_page: Items per page, default 100
        effective_score_min: Min effective score (1-5)
        effective_score_max: Max effective score (1-5)
        comparisons_min: Min comparison count (0-10)
        comparisons_max: Max comparison count (0-10)
        volatility_min: Min volatility (0-1)
        volatility_max: Max volatility (0-1)
    """
    from .cache import get_scored
    from pathlib import Path

    try:
        # Pagination parameters
        page = request.args.get("page", default=1, type=int)
        per_page = request.args.get("per_page", default=100, type=int)
        if page < 1:
            page = 1
        if per_page < 1:
            per_page = 100

        offset = (page - 1) * per_page

        # Filter parameters (optional)
        effective_score_min = request.args.get(
            "effective_score_min", default=None, type=float
        )
        effective_score_max = request.args.get(
            "effective_score_max", default=None, type=float
        )
        comparisons_min = request.args.get("comparisons_min", default=None, type=int)
        comparisons_max = request.args.get("comparisons_max", default=None, type=int)
        volatility_min = request.args.get("volatility_min", default=None, type=float)
        volatility_max = request.args.get("volatility_max", default=None, type=float)

        # Query DB for scored images with pagination and filters
        rows, total = get_scored(
            limit=per_page,
            offset=offset,
            effective_score_min=effective_score_min,
            effective_score_max=effective_score_max,
            comparisons_min=comparisons_min,
            comparisons_max=comparisons_max,
            volatility_min=volatility_min,
            volatility_max=volatility_max,
        )

        root = image_root()
        root_path = Path(root)

        scores: list[Any] = []
        for r in rows:
            try:
                path: str = str(r.get("path"))
                # Convert absolute path to relative (forward slashes)
                try:
                    rel = Path(path).relative_to(root_path).as_posix()
                except Exception:
                    rel = Path(path).as_posix()

                scores.append(
                    {
                        "file_id": rel,
                        "image": rel,
                        "score": r.get("score"),
                        "modifier": r.get("score_modifier", 0.0),
                        "comparison_count": r.get("comparison_count", 0),
                        "volatility": r.get("volatility", 0.0),
                    }
                )
            except Exception as e:
                print(f"Error processing row {r}: {e}")
                continue

        return jsonify(
            {"scores": scores, "total": total, "page": page, "per_page": per_page}
        )
    except Exception as e:
        print(f"Error getting scores: {e}")
        return jsonify({"error": str(e), "scores": [], "total": 0}), 500


# ───────────────────────────────
# Random unscored endpoints
# ───────────────────────────────


@app.route("/random_unscored")
def random_unscored():
    unscored = get_unscored_images(image_root())
    if not unscored:
        return "", 204
    img = random.choice(unscored)
    return jsonify({"image": img})


@app.route("/random_unscored_batch")
def random_unscored_batch():
    unscored = get_unscored_images(image_root())
    if not unscored:
        return ("", 204)
    arg_val = request.args.get("n")
    if arg_val is None:
        return ("missing n", 400)
    n = int(arg_val)
    if n < 1:
        return ("invalid n", 400)
    if n > len(unscored):
        n = len(unscored)
    choices = random.sample(unscored, n)
    return jsonify({"images": choices})


# ───────────────────────────────
# Score submission endpoints
# ───────────────────────────────
@app.route("/submit_score", methods=["POST"])
def submit_score_route():
    return submit_scores_handler(request.json)


@app.route("/submit_scores", methods=["POST"])
def submit_scores_route():
    return submit_scores_handler(request.json)


# ───────────────────────────────
# Comparison mode endpoints
# ───────────────────────────────


@app.route("/compare/next/")
def compare_next():
    """
    Get next pair of images with same score for comparison.

    Returns:
        JSON with {left: {image, score, ..}, right: {image, score, ..}}
        or 204 if no pairs available
    """
    root = image_root()

    score: int = request.args.get("score", type=int, default=0)
    # 0 means any score

    pair_data = get_paired_images(
        score, safety_limit=100, max_comparison_count=10, max_tolerance=1.5
    )

    if not pair_data:
        return (
            "{}",
            200,
        )  # empty response to indicate no pairs available, but not an error

    img1_path, img2_path, img1_data, img2_data = pair_data

    # Convert absolute paths to relative for client
    root_path = Path(root)
    try:
        rel_path1 = str(Path(img1_path).relative_to(root_path))
        rel_path2 = str(Path(img2_path).relative_to(root_path))
    except ValueError:
        rel_path1: str = img1_path
        rel_path2: str = img2_path

    return jsonify(
        {
            "left": {
                "image": rel_path1,
                "score": img1_data["score"],
                "comparison_count": img1_data["comparison_count"],
                "score_modifier": img1_data["score_modifier"],
                "volatility": img1_data["volatility"],
            },
            "right": {
                "image": rel_path2,
                "score": img2_data["score"],
                "comparison_count": img2_data["comparison_count"],
                "score_modifier": img2_data["score_modifier"],
                "volatility": img2_data["volatility"],
            },
        }
    )


@app.route("/compare/submit", methods=["POST"])
def compare_submit():
    """
    Submit comparison result and update both images' scores.

    Expected JSON:
    {
        winner_image: str (relative path),
        loser_image: str (relative path),
        winner_data: {...},
        loser_data: {...},
    }
    """
    data = request.json
    if not data:
        return jsonify({"ok": False, "error": "Missing data"}), 400

    winner_rel = data.get("winner_image")
    loser_rel = data.get("loser_image")
    winner_data = data.get("winner_data", {})
    loser_data = data.get("loser_data", {})

    if not winner_rel or not loser_rel:
        return (
            jsonify({"ok": False, "error": "Missing winner or loser image paths"}),
            400,
        )

    # Convert relative paths to absolute
    root = image_root()
    root_path = Path(root)

    try:
        winner_path = str(root_path / winner_rel)
        loser_path = str(root_path / loser_rel)
    except Exception as e:
        return jsonify({"ok": False, "error": f"Path error: {str(e)}"}), 400

    # Apply comparison logic
    try:
        winner_data, loser_data, success, err = apply_comparison_and_write(
            winner_data, winner_path, loser_data, loser_path
        )
        if not success:
            return (
                jsonify({"ok": False, "error": f"Error updating comparison: {err}"}),
                500,
            )

        return jsonify(
            {
                "ok": True,
                "winner_score": winner_data["score"],
                "loser_score": loser_data["score"],
                "winner_count": winner_data["comparison_count"],
                "loser_count": loser_data["comparison_count"],
            }
        )
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ───────────────────────────────
# Run server
# ───────────────────────────────
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--test-run", action="store_true")
    args = parser.parse_args()

    if args.test_run:
        try:
            r = image_root()
            print(f"Image root: {r}")
            print("Configuration looks good.")
            sys.exit(0)
        except Exception as e:
            print(f"Configuration failed: {e}")
            sys.exit(1)
    else:
        app.run(host="0.0.0.0", port=5001, debug=True)
