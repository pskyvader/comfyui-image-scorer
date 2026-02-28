import sys
from pathlib import Path
import random
from threading import Thread
from time import sleep
from flask import Flask, request, send_from_directory, jsonify

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
from .comparison import (
    get_paired_images,
    apply_comparison,
    write_comparison_data,
)
from .cache import (
    get_absolute_total,
    total_cached,
    is_scanning,
    is_finished,
    start_scan,
    finish_scan,
)

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent

# ───────────────────────────────
# Background scanning worker
# ───────────────────────────────
_scan_thread = None
_stop_scan = False


@app.teardown_request
def log_on_teardown(exception=None):
    if exception:
        print(f"CRITICAL: The route '{request.path}' crashed with: {exception}")


# @app.errorhandler(Exception)
# def handle_exception(e):
#     print(f"CRITICAL: An unhandled exception occurred: {e}")
#     return jsonify({"error": "Internal server error"}), 500


def background_scan(batch_size: int = 1000):
    """
    Continuously scan images in batches until all are processed.
    Updates the cache and ensures valid/unscored counts are accurate.
    """
    global _stop_scan
    start_scan()
    retries = 0

    root = image_root()
    while not _stop_scan:
        added = scan_batch(root, limit=batch_size)
        if not added:
            if retries > 3:
                _stop_scan = True
            sleep(0.1)  # wait a bit if no new images
            valid_images = get_unscored_images(root)
            if len(valid_images) > 0:
                retries += 1
        else:
            retries = 0
            sleep(0.5)  # small delay for incremental updates
    finish_scan()


def trigger_scan():
    """Start the background scan if not running."""
    global _scan_thread, _stop_scan
    if _scan_thread is None or not _scan_thread.is_alive():
        _stop_scan = False
        _scan_thread = Thread(target=background_scan, daemon=True)
        _scan_thread.start()


# ───────────────────────────────
# Static file routes
# ───────────────────────────────
@app.route("/")
def serve_index():
    return send_from_directory(str(BASE_DIR), "index.html")


@app.route("/style.css")
def serve_css():
    return send_from_directory(str(BASE_DIR), "style.css")


@app.route("/client.js")
def serve_js():
    return send_from_directory(str(BASE_DIR), "client.js")


@app.route("/image/<path:subpath>")
def serve_image_route(subpath: str):
    return serve_file(subpath)


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

    valid_images = get_unscored_images(image_root())
    valid = len(valid_images)
    cached_unscored = total_cached_unscored()  # Unscored images in cache
    cached_total = total_cached()  # All images in cache
    total = get_absolute_total()

    if is_scanning() and valid == 0:
        state = "scanning"
    elif valid > 0:
        state = "ready"
    elif is_finished():
        state = "done"
    else:
        state = "empty"

    # Get comparison mode stats
    comp_stats = get_comparison_stats()

    return jsonify(
        {
            "state": state,
            "valid": valid,
            "cached_unscored": cached_unscored,
            "cached_total": cached_total,
            "total": total,
            "comparison": comp_stats,
        }
    )


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

    score = request.args.get("score", type=int)
    if score is None:
        score = 0  # 0 means any score

    pair_data = get_paired_images(score)

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
        rel_path1 = img1_path
        rel_path2 = img2_path

    return jsonify(
        {
            "left": {
                "image": rel_path1,
                "score": img1_data.get("score", 3),
                "comparison_count": img1_data.get("comparison_count", 0),
                "score_modifier": img1_data.get("score_modifier", 0),
            },
            "right": {
                "image": rel_path2,
                "score": img2_data.get("score", 3),
                "comparison_count": img2_data.get("comparison_count", 0),
                "score_modifier": img2_data.get("score_modifier", 0),
            },
        }
    )


@app.route("/compare/submit", methods=["POST"])
def compare_submit():
    """
    Submit comparison result and update both images' scores.

    Expected JSON:
    {
        winner: "left" | "right",
        left_image: str (relative path),
        right_image: str (relative path),
        left_data: {...},
        right_data: {...},
    }
    """
    data = request.json
    if not data:
        return jsonify({"ok": False, "error": "Missing data"}), 400

    winner = data.get("winner")
    left_rel = data.get("left_image")
    right_rel = data.get("right_image")
    left_data = data.get("left_data", {})
    right_data = data.get("right_data", {})

    if not winner or not left_rel or not right_rel:
        return jsonify({"ok": False, "error": "Missing winner or image paths"}), 400

    # Convert relative paths to absolute
    root = image_root()
    root_path = Path(root)

    try:
        left_abs = str(root_path / left_rel)
        right_abs = str(root_path / right_rel)
    except Exception as e:
        return jsonify({"ok": False, "error": f"Path error: {str(e)}"}), 400

    # Determine winner and loser
    if winner == "left":
        winner_path = left_abs
        loser_path = right_abs
        winner_data = left_data.copy()
        loser_data = right_data.copy()
    elif winner == "right":
        winner_path = right_abs
        loser_path = left_abs
        winner_data = right_data.copy()
        loser_data = left_data.copy()
    else:
        return jsonify({"ok": False, "error": "Invalid winner value"}), 400

    # Apply comparison logic
    try:
        winner_data, loser_data = apply_comparison(winner_data, loser_data)

        # Write updated data
        success, err = write_comparison_data(
            winner_path, loser_path, winner_data, loser_data
        )

        if not success:
            return jsonify({"ok": False, "error": err}), 500

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
