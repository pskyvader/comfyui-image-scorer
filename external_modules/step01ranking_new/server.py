"""Main server - Flask application for ranking system."""

import sys
import threading
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
        __package__ = "external_modules.step01ranking_new"


# Now import Flask and API modules

from api.ranking_api_v2 import register_ranking_routes
from api.gallery_api import register_gallery_routes
from file_management.folder_organizer import ensure_tier_structure
from file_management.path_handler import get_ranked_root, compute_path_from_filename, find_image_path
from database.images_table import get_image as get_db_image
from image_processor import ImageProcessor
from shared.config import config
from shared.paths import image_root

# Global image processor
image_processor = ImageProcessor()

app = Flask(__name__, static_folder=None)
app.extensions["image_processor"] = image_processor
setattr(app, "image_processor", image_processor)  # Also set as attribute for easy access
logger = app.logger

# Set up Flask configuration
app.config["JSON_SORT_KEYS"] = False

# Register API routes FIRST before any other routes
logger.debug("Registering API blueprints...")
register_ranking_routes(app)
logger.debug("Ranking routes registered")
register_gallery_routes(app)
logger.debug("Gallery routes registered")

# Global image processor state
scanner_thread: threading.Thread | None = None


def scanner_task(img_root: str) -> None:
    image_processor.process_next_batch(img_root)
    logger.info("[SCANNER] Initialization batch complete.")


def start_background_scanner(img_root: str) -> None:
    """Start background thread to perform global image initialization."""
    global scanner_thread
    scanner_thread = threading.Thread(
        target=scanner_task, daemon=True, args=(img_root,)
    )
    scanner_thread.start()
    logger.info("[SCANNER] Global image scanner started.")


def startup_worker(img_root: str, sync_existing: bool = False) -> None:
    """Background worker for initialization tasks."""
    # Step 1: Initialize score folders
    logger.info("[1/3] Initializing score folder structure...")
    if not ensure_tier_structure():
        logger.error("Failed to initialize folder structure")
        return
    logger.info("  [OK] Folder structure ready")

    # Step 2: Rebuild database from existing ranked images
    if sync_existing:
        logger.info("[2/3] Rebuilding database from existing ranked images (MANUAL SYNC ENABLED)...")
        image_processor.rebuild_database_from_ranked()
        logger.info("  [OK] Database rebuild complete")
    else:
        logger.info("[2/3] Skipping automatic database rebuild (use --sync-existing to enable)...")

    # Step 3: Start background scanner for new images
    if img_root:
        logger.info(f"[3/3] Starting background image scanner (root: {img_root})...")
        # We call it directly here as we are already in a background thread
        scanner_task(img_root)
    else:
        logger.info("[3/3] Skipping scanner (no image_root configured)")


def init_ranking_system(img_root: str | None = None, sync_existing: bool = False) -> bool:
    """Initialize system and trigger background recovery."""
    logger.info("\n" + "=" * 60)
    logger.info("RANKING SYSTEM INITIALIZATION (BACKGROUND)")
    logger.info("=" * 60)

    # Trigger background worker for all heavy tasks
    threading.Thread(target=startup_worker, args=(img_root, sync_existing), daemon=True).start()

    logger.info("[OK] Background initialization triggered.")
    logger.info("=" * 60)
    return True


# Static file routes
@app.route("/")
def serve_index():
    """Serve main index page."""
    base_dir = Path(__file__).parent
    return send_from_directory(str(base_dir / "frontend" / "html"), "index.html")


@app.route("/css/<path:filename>")
def serve_css(filename: str):
    """Serve CSS files."""
    base_dir = Path(__file__).parent
    return send_from_directory(str(base_dir / "frontend" / "css"), filename)


@app.route("/js/<path:filename>")
def serve_js(filename: str):
    """Serve JavaScript files."""
    base_dir = Path(__file__).parent
    return send_from_directory(str(base_dir / "frontend" / "js"), filename)


@app.route("/output/ranked/<path:filepath>")
def serve_ranked_image(filepath: str):
    """Serve images from scored folders."""
    ranked_root = get_ranked_root()

    # Decode any percent-encoding from the client
    filepath_decoded = unquote(filepath)

    # If direct path exists, serve it
    direct_path = ranked_root / filepath_decoded
    if direct_path.exists() and direct_path.is_file():
        logger.debug(f"Found image at direct path: {direct_path}")
        response = send_file(str(direct_path))
        # Disable caching
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    # Otherwise search recursively for the filename (handles nested subfolders)
    filename = Path(filepath_decoded).name
    found = find_image_path(filename)
    if found:
        found_path = Path(found)
        logger.debug(
            f"Serving found image: {found_path} (root:{str(ranked_root)}) (parent: {found_path.parent})"
        )
        response = send_file(str(found_path))
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    # Not found
    logger.warning(f"Image not found in ranked folders: {filepath}")
    return {"error": "Image not found"}, 404


@app.route("/images/<path:filename>")
def serve_image_by_name(filename: str):
    """Serve an image by filename. Looks up by provided score query, DB, or recursive search."""
    ranked_root = get_ranked_root()
    fname = Path(unquote(filename)).name

    # Prefer explicit score query param when provided
    score_q = request.args.get("score", type=float)

    # 1) If score provided, compute the expected path and try to serve
    if score_q is not None:
        dest = compute_path_from_filename(fname, score_q)
        if dest.exists() and dest.is_file():
            return send_file(str(dest))

    # 2) Try DB score
    db_entry = get_db_image(fname)
    if db_entry and db_entry.get("score") is not None:
        dest = compute_path_from_filename(fname, db_entry["score"])
        if dest.exists() and dest.is_file():
            return send_file(str(dest))

    # 3) Fallback: recursive search
    found = find_image_path(fname)
    if found:
        return send_file(str(found))

    return {"error": "Image not found"}, 404


@app.route("/image/<path:filename>")
def serve_image_alias(filename: str):
    """Compatibility alias for legacy frontends that use `/image/`."""
    return serve_image_by_name(filename)


@app.route("/api/<path:path>")
def catch_api_404(path: str):
    """Explicitly handle unmapped API paths."""
    return {"error": f"API endpoint not found: /api/{path}"}, 404


@app.route("/<path:filename>")
def serve_html(filename: str):
    """Serve HTML files - only matches files in frontend/html."""
    base_dir = Path(__file__).parent
    return send_from_directory(str(base_dir / "frontend" / "html"), filename)


# Error handlers
@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    return {"error": "Not found"}, 404


@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors."""
    logger.error(f"Server error: {e}")
    return {"error": "Server error"}, 500


# Debug: Print all registered routes
logger.debug("Registered routes:")
for rule in app.url_map.iter_rules():
    logger.debug(f"  {rule.rule} -> {rule.endpoint}")


def main():
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
        format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s',
        datefmt='%H:%M:%S',
        force=True # Ensure we override any existing configuration
    )

    # Initialize system
    # If debug mode is ON, Flask runs twice (reloader).
    # We only want to initialize in the actual worker process.
    should_init = True
    if args.debug and os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        should_init = False

    if should_init:
        if not init_ranking_system(args.image_root, sync_existing=args.sync_existing):
            logger.error("Failed to initialize system")
            return 1

    # Start server
    logger.info(f"Starting ranking server on {args.host}:{args.port}...\n")

    try:
        app.run(host=args.host, port=args.port, debug=args.debug)
    except KeyboardInterrupt:
        logger.info("\n\nServer stopped")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
