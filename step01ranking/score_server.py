import sys
from pathlib import Path
import random
from threading import Thread
from time import sleep
from flask import Flask, request, send_from_directory, jsonify

# Add project root for absolute imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from step01ranking.utils import (
    get_unscored_images,
    serve_file,
    image_root,
    scan_batch,
)
from step01ranking.scores import submit_scores_handler
from step01ranking.cache import (
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


def background_scan(batch_size: int = 1000):
    """
    Continuously scan images in batches until all are processed.
    Updates the cache and ensures valid/unscored counts are accurate.
    """
    global _stop_scan
    start_scan()
    retries=0
        
    root = image_root()
    while not _stop_scan:
        added = scan_batch(root, limit=batch_size)
        if not added:
            if retries>3:
                _stop_scan=True
            sleep(0.1)  # wait a bit if no new images
            valid_images = get_unscored_images(image_root())
            if len(valid_images)>0:
                retries+=1
        else:
            retries=0
            sleep(0.5)  # small delay for incremental updates
    finish_scan()


def trigger_scan():
    """Start the background scan if not running."""
    global _scan_thread, _stop_scan
    if _scan_thread is None or not _scan_thread.is_alive():
        _stop_scan=False
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
    
    trigger_scan()

    valid_images = get_unscored_images(image_root())
    valid = len(valid_images)
    cached = total_cached()
    total = get_absolute_total()

    if is_scanning() and valid == 0:
        state = "scanning"
    
    elif valid > 0:
        state = "ready"
        
    elif is_finished():
        state = "done"
    else:
        state = "empty"


    return jsonify({"state": state, "valid": valid, "cached": cached, "total": total})


# ───────────────────────────────
# Random unscored endpoints
# ───────────────────────────────


# @app.route("/random_unscored")
# def random_unscored():
#     unscored = get_unscored_images(image_root())
#     if not unscored:
#         return ("", 204)
#     choice = random.choice(unscored)
#     return jsonify({"image": choice})


@app.route("/random_unscored")
def random_unscored():
    unscored = get_unscored_images(image_root())
    if not unscored:
        return "", 204

    # Pick a random image and fix slashes for browser
    img = random.choice(unscored)#.replace("\\", "/")
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


# @app.route("/random_unscored_batch")
# def random_unscored_batch():
#     unscored = get_unscored_images(image_root())
#     if not unscored:
#         return "", 204

#     n = request.args.get("n", type=int)
#     if n is None or n < 1:
#         return "invalid n", 400
#     if n > len(unscored):
#         n = len(unscored)

#     # Convert all paths to browser-friendly slashes
#     imgs = [img.replace("\\", "/") for img in random.sample(unscored, n)]
#     return jsonify({"images": imgs})


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
# Run server
# ───────────────────────────────
if __name__ == "__main__":
    import argparse

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
