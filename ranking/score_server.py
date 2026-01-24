import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import random
from typing import Any, List, Tuple

from flask import Flask, request, send_from_directory, jsonify
from shared.config import PROJECT_ROOT, config
from ranking.utils import get_unscored_images, serve_file
from ranking.scores import submit_scores_handler

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent


def image_root() -> str:
    img_root = config["image_root"]
    root_path = Path(img_root)
    if not root_path.is_absolute():
        root_path = PROJECT_ROOT.joinpath(root_path).resolve()
    return str(root_path)


def root_and_unscored() -> Tuple[str, List[str]]:
    root = image_root()
    return root, get_unscored_images(root)


@app.route("/")
def serve_index():
    return send_from_directory(str(BASE_DIR), "index.html")


@app.route("/style.css")
def serve_css():
    return send_from_directory(str(BASE_DIR), "style.css")


@app.route("/image/<path:subpath>")
def serve_image_route(subpath: str) -> Any:
    return serve_file(subpath)


@app.route("/unscored_count")
def unscored_count() -> Any:
    root, unscored = root_and_unscored()
    return jsonify({"root": root, "count": len(unscored)})


@app.route("/random_unscored")
def random_unscored() -> Any:
    _, unscored = root_and_unscored()
    if not unscored:
        return ("", 204)
    choice = random.choice(unscored)
    return jsonify({"image": choice})


@app.route("/random_unscored_batch")
def random_unscored_batch() -> Any:
    _, unscored = root_and_unscored()
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


@app.route("/submit_scores", methods=["POST"])
def submit_scores_route() -> Any:
    return submit_scores_handler(request.json)


@app.route("/metadata/<path:subpath>")
def serve_metadata_route(subpath: str) -> Any:
    return serve_file(subpath)


@app.route("/submit_score", methods=["POST"])
def submit_score_route() -> Any:
    return submit_scores_handler(request.json)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-run", action="store_true", help="Verify configuration and exit")
    args = parser.parse_args()

    if args.test_run:
        print("Verifying configuration...")
        try:
            # Check if critical paths exist/throw
            r = image_root()
            print(f"Image root resolves to: {r}")
            print("Configuration looks good.")
            sys.exit(0)
        except Exception as e:
            print(f"Configuration failed: {e}")
            sys.exit(1)

    app.run(host="0.0.0.0", port=5001, debug=True)
