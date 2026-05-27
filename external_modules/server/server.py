"""Main server - Flask application for ranking system."""

import sys
import threading
import time
import os
from pathlib import Path
from flask import Flask, send_from_directory, request, send_file
from urllib.parse import unquote
import argparse
import logging

# Set up paths FIRST before any imports
current_dir = str(Path(__file__).parent)
root_path = str(Path(__file__).parents[2])  # comfyui-image-scorer
if root_path not in sys.path:
    sys.path.insert(0, root_path)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if __name__ == "__main__":
    if __package__ is None:
        __package__ = "external_modules.server"


# Now import Flask and API modules

from external_modules.comparison.endpoints import register_ranking_routes
from external_modules.gallery.endpoints import register_gallery_routes
from external_modules.maps.endpoints import register_maps_routes
from external_modules.database_structure.endpoints import register_database_routes
from external_modules.data_transform.endpoints import register_data_transform_routes
from external_modules.training_hyperparameters.endpoints import register_training_routes
from external_modules.analysis.endpoints import register_analysis_routes
from external_modules.database_structure.folder_organizer import ensure_tier_structure
from external_modules.database_structure.path_handler import (
    get_ranked_root,
    compute_path_from_filename,
    find_image_path,
)
from external_modules.database_structure.images_table import get_image as get_db_image
from .image_processor import ImageProcessor

# from shared.config import config
from shared.paths import image_root
logger = logging.getLogger(__name__)

# Global image processor
image_processor = ImageProcessor()

app = Flask(__name__, static_folder=None)
app.extensions["image_processor"] = image_processor
setattr(
    app, "image_processor", image_processor
)  # Also set as attribute for easy access
logger: logging.Logger = app.logger

# Set up Flask configuration
app.config["JSON_SORT_KEYS"] = False

# Register API routes FIRST before any other routes
register_ranking_routes(app)

register_gallery_routes(app)
register_maps_routes(app)

register_database_routes(app)
register_data_transform_routes(app)
register_training_routes(app)
register_analysis_routes(app)

# Global image processor state
scanner_thread: threading.Thread | None = None


def scanner_task(str) -> None:
    sleep_time = 30  # Start with 30s, then back off if no new images found
    while True:
        stats = image_processor.process_next_batch(img_root, batch_size=100)
        added = stats["added"]
        if added > 0:
            sleep_time = 30
        else:
            sleep_time *= 2
            sleep_time = min(sleep_time, 600)

        logger.info(f"Added:{added}, Sleeping {sleep_time}s...")
        time.sleep(sleep_time)


def start_background_scanner(str) -> None:
    """Start background thread to perform global image initialization."""
    global scanner_thread
    if scanner_thread:
        logger.info(
            f"[SCANNER] Scanner already running. Alive: {scanner_thread.is_alive()}"
        )
        logger.debug("start_background_scanner took %.4fs", time.perf_counter() - _start)
        return
    scanner_thread = threading.Thread(
        target=scanner_task, daemon=True, args=(img_root,)
    )
    scanner_thread.start()
    logger.info("[SCANNER] Global image scanner started.")


def startup_worker(str, sync_existing: bool = False) -> None:
    """Background worker for initialization tasks."""
    # Step 1: Initialize score folders
    if not ensure_tier_structure():
        logger.debug("startup_worker took %.4fs", time.perf_counter() - _start)
        return

    # Step 2: Rebuild database from existing ranked images
    if sync_existing:
        image_processor.rebuild_database_from_ranked()

    # Step 3: Start background scanner for new images
    if img_root:
        # We call it directly here as we are already in a background thread
        scanner_task(img_root)


def init_ranking_system(
    _start = time.perf_counter()
    _start = time.perf_counter()
    img_root: str | None = None, sync_existing: bool = False
    logger.debug("init_ranking_system took %.4fs", time.perf_counter() - _start)
) -> bool:
    """Initialize system and trigger background recovery."""

    # Trigger background worker for all heavy tasks
    threading.Thread(
        target=startup_worker, args=(img_root, sync_existing), daemon=True
    ).start()

    logger.info("[OK] Background initialization triggered.")
    return True


# Static file routes
SECTION_FRONTENDS = {
    "comparison": Path(__file__).parent.parent / "comparison" / "frontend",
    "gallery": Path(__file__).parent.parent / "gallery" / "frontend",
    "maps": Path(__file__).parent.parent / "maps" / "frontend",
    "database": Path(__file__).parent.parent / "database_structure" / "frontend",
    "data": Path(__file__).parent.parent / "data_transform" / "frontend",
    "training": Path(__file__).parent.parent / "training_hyperparameters" / "frontend",
    "analysis": Path(__file__).parent.parent / "analysis" / "frontend",
}

SERVER_FRONTEND = Path(__file__).parent / "frontend"


@app.route("/")
def serve_index():
    """Serve main index page."""
    return send_from_directory(str(SERVER_FRONTEND / "html"), "index.html")


    _start = time.perf_counter()
    _start = time.perf_counter()
@app.route("/css/<path:filename>")
logger.debug("serve_index took %.4fs", time.perf_counter() - _start)
logger.debug("serve_index took %.4fs", time.perf_counter() - _start)
def serve_css(str):
    """Serve CSS files (legacy - from server frontend)."""
    result = send_from_directory(str(SERVER_FRONTEND / "css"), filename)
    logger.debug("serve_css took %.4fs", time.perf_counter() - _start)
    return result


@app.route("/js/<path:filename>")
def serve_js(str):
    """Serve JavaScript files (legacy - from server frontend)."""
    result = send_from_directory(str(SERVER_FRONTEND / "js"), filename)
    logger.debug("serve_js took %.4fs", time.perf_counter() - _start)
    return result


@app.route("/static/<section>/<path:filename>")
def serve_section_static(str, filename: str):
    """Serve static files from a section's frontend/ folder."""
    if section not in SECTION_FRONTENDS:
        result = {"error": f"Unknown section: {section}"}, 404
        logger.debug("serve_section_static took %.4fs", time.perf_counter() - _start)
        return result
    result = send_from_directory(str(SECTION_FRONTENDS[section]), filename)
    logger.debug("serve_section_static took %.4fs", time.perf_counter() - _start)
    return result


@app.route("/output/ranked/<path:filepath>")
def serve_ranked_image(str):
    """Serve images from scored folders."""
    ranked_root = get_ranked_root()

    # Decode any percent-encoding from the client
    filepath_decoded = unquote(filepath)

    # If direct path exists, serve it
    direct_path = ranked_root / filepath_decoded
    if direct_path.exists() and direct_path.is_file():
        response = send_file(str(direct_path))
        # Disable caching
        response.headers["Cache-Control"] = (
            "no-store, no-cache, must-revalidate, max-age=0"
        )
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        result = response
        logger.debug("serve_ranked_image took %.4fs", time.perf_counter() - _start)
        return result

    # Otherwise search recursively for the filename (handles nested subfolders)
    filename = Path(filepath_decoded).name
    found = find_image_path(filename)
    if found:
        found_path = Path(found)
        response = send_file(str(found_path))
        response.headers["Cache-Control"] = (
            "no-store, no-cache, must-revalidate, max-age=0"
        )
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        result = response
        logger.debug("serve_ranked_image took %.4fs", time.perf_counter() - _start)
        return result

    # Not found
    logger.warning(f"Image not found in ranked folders: {filepath}")
    result = {"error": "Image not found"}, 404
    logger.debug("serve_ranked_image took %.4fs", time.perf_counter() - _start)
    return result


@app.route("/images/<path:filename>")
def serve_image_by_name(str):
    """Serve an image by filename. Looks up by provided score query, DB, or recursive search."""
    ranked_root = get_ranked_root()
    fname = Path(unquote(filename)).name

    # Prefer explicit score query param when provided
    score_q = request.args.get("score", type=float)

    # 1) If score provided, compute the expected path and try to serve
    if score_q is not None:
        dest = compute_path_from_filename(fname, score_q)
        if dest.exists() and dest.is_file():
            result = send_file(str(dest))
            logger.debug("serve_image_by_name took %.4fs", time.perf_counter() - _start)
            return result

    # 2) Try DB score
    db_entry = get_db_image(fname)
    if db_entry and db_entry.get("score") is not None:
        dest = compute_path_from_filename(fname, db_entry["score"])
        if dest.exists() and dest.is_file():
            result = send_file(str(dest))
            logger.debug("serve_image_by_name took %.4fs", time.perf_counter() - _start)
            return result

    # 3) Fallback: recursive search
    found = find_image_path(fname)
    if found:
        result = send_file(str(found))
        logger.debug("serve_image_by_name took %.4fs", time.perf_counter() - _start)
        return result

    result = {"error": "Image not found"}, 404
    logger.debug("serve_image_by_name took %.4fs", time.perf_counter() - _start)
    return result


@app.route("/image/<path:filename>")
def serve_image_alias(str):
    """Compatibility alias for legacy frontends that use `/image/`."""
    result = serve_image_by_name(filename)
    logger.debug("serve_image_alias took %.4fs", time.perf_counter() - _start)
    return result


@app.route("/api/<path:path>")
def catch_api_404(str):
    """Explicitly handle unmapped API paths."""
    result = {"error": f"API endpoint not found: /api/{path}"}, 404
    logger.debug("catch_api_404 took %.4fs", time.perf_counter() - _start)
    return result


@app.route("/<path:filename>")
def serve_html(str):
    """Serve HTML files - only matches files in frontend/html."""
    base_dir = Path(__file__).parent
    result = send_from_directory(str(base_dir / "frontend" / "html"), filename)
    logger.debug("serve_html took %.4fs", time.perf_counter() - _start)
    return result


# Error handlers
@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    return {"error": "Not found"}, 404


    _start = time.perf_counter()
    _start = time.perf_counter()
@app.errorhandler(500)
logger.debug("not_found took %.4fs", time.perf_counter() - _start)
logger.debug("not_found took %.4fs", time.perf_counter() - _start)
def server_error(e):
    """Handle 500 errors."""
    logger.error(f"Server error: {e}")
    return {"error": "Server error"}, 500


    _start = time.perf_counter()
    _start = time.perf_counter()
def main():
logger.debug("server_error took %.4fs", time.perf_counter() - _start)
    _start = time.perf_counter()
logger.debug("server_error took %.4fs", time.perf_counter() - _start)
    """Main entry point."""

    parser = argparse.ArgumentParser(description="Ranking System Server")
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5001,
        help="Port to bind to (default: 5001)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )
    parser.add_argument(
        "--image-root",
        default=image_root,
        help="Root folder for images (for migration)",
    )
    parser.add_argument(
        "--sync-existing",
        action="store_true",
        help="Rebuild database from existing ranked images on startup",
    )

    args = parser.parse_args()

    # Configure global logging based on debug flag
    log_level = logging.DEBUG if args.debug else logging.INFO

    # Simple configuration that works well for both console and redirected output
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
        force=True,  # Ensure we override any existing configuration
    )

    # Initialize system
    # If debug mode is ON, Flask runs twice (reloader).
    # We only want to initialize in the actual worker process.
    should_init = True
    if args.debug and os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        should_init = False

    if should_init:
        if not init_ranking_system(args.image_root, sync_existing=args.sync_existing):
            return 1

    # Start server
    logger.info(f"Starting ranking server on {args.host}:{args.port}...\n")

    try:
        app.run(host=args.host, port=args.port, debug=args.debug)
    except KeyboardInterrupt:
        return 0

    return 0


    _start = time.perf_counter()
if __name__ == "__main__":
logger.debug("main took %.4fs", time.perf_counter() - _start)
    sys.exit(main())
