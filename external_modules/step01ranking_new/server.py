"""Main server - Flask application for ranking system."""

import sys
import threading
import os
from pathlib import Path
from time import sleep

# Set up paths FIRST before any imports
if __name__ == "__main__":
    root_path = str(Path(__file__).parent.parent.parent)  # comfyui-image-scorer
    sys.path.insert(0, root_path)
    if __package__ is None:
        __package__ = "external_modules.step01ranking_new"
else:
    # When imported as module
    root_path = str(Path(__file__).parent.parent.parent)  # comfyui-image-scorer
    if root_path not in sys.path:
        sys.path.insert(0, root_path)

# Now import Flask and API modules
from flask import Flask, send_from_directory, request, send_file
from urllib.parse import unquote
from api.ranking_api_v2 import register_ranking_routes
from api.gallery_api import register_gallery_routes
from file_management.folder_organizer import ensure_tier_structure
from image_processor import ImageProcessor
from shared.config import config
from shared.paths import image_root

# Global image processor
image_processor = ImageProcessor()

app = Flask(__name__, static_folder=None)
app.image_processor = image_processor
logger = app.logger

# Set up Flask configuration
app.config["JSON_SORT_KEYS"] = False

# Register API routes FIRST before any other routes
try:
    print("[DEBUG] Registering API blueprints...")
    register_ranking_routes(app)
    print("[DEBUG] Ranking routes registered")
    register_gallery_routes(app)
    print("[DEBUG] Gallery routes registered")
except Exception as e:
    print(f"[ERROR] Failed to register API routes: {e}")
    import traceback
    traceback.print_exc()

# Global image processor state
scanner_thread = None

def scanner_task(image_root: str):
    global image_processor
    try:
        image_processor.process_next_batch(image_root)
        print("[SCANNER] Initialization batch complete.")
    except Exception as e:
        print(f"[SCANNER] Error: {e}")


def start_background_scanner(image_root: str):
    """Start background thread to perform global image initialization."""
    global scanner_thread
    scanner_thread = threading.Thread(
        target=scanner_task, daemon=True, args=(image_root,)
    )
    scanner_thread.start()
    print("[SCANNER] Global image scanner started.")


def startup_worker(image_root: str):
    """Background worker for initialization tasks."""
    try:
        # Step 1: Initialize score folders
        print("[1/3] Initializing score folder structure...")
        if not ensure_tier_structure():
            print("ERROR: Failed to initialize folder structure")
            return
        print("  [OK] Folder structure ready")

        # Step 2: Rebuild database from existing ranked images (for recovery from DB loss)
        print("[2/3] Rebuilding database from existing ranked images...")
        processor = ImageProcessor()
        processor.rebuild_database_from_ranked()
        print("  [OK] Database rebuild started/complete")

        # Step 3: Start background scanner for new images
        if image_root:
            print(f"[3/3] Starting background image scanner (root: {image_root})...")
            # We call it directly here as we are already in a background thread
            scanner_task(image_root)
        else:
            print("[3/3] Skipping scanner (no image_root configured)")
    except Exception as e:
        print(f"ERROR during background startup: {e}")


def init_ranking_system(image_root: str = None) -> bool:
    """Initialize system and trigger background recovery."""
    print("\n" + "=" * 60)
    print("RANKING SYSTEM INITIALIZATION (BACKGROUND)")
    print("=" * 60)
    
    # Trigger background worker for all heavy tasks
    import threading
    threading.Thread(target=startup_worker, args=(image_root,), daemon=True).start()
    
    print("[OK] Background initialization triggered.")
    print("=" * 60)
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
    from file_management.path_handler import get_ranked_root, find_image_path
    from pathlib import Path

    ranked_root = get_ranked_root()

    # Decode any percent-encoding from the client
    try:
        filepath_decoded = unquote(filepath)
    except Exception:
        filepath_decoded = filepath

    # If direct path exists, serve it
    direct_path = ranked_root / filepath_decoded
    #logger.debug(f"Attempting to serve ranked image: {direct_path}")
    if direct_path.exists() and direct_path.is_file():
        logger.debug(f"Found image at direct path: {direct_path}")
        return send_file(str(direct_path))

    # Otherwise search recursively for the filename (handles nested subfolders)
    filename = Path(filepath_decoded).name
    found = find_image_path(filename)
    #logger.debug(f"Recursive search for '{filename}' found: {found}")
    if found:
        found_path = Path(found)
        try:
            logger.debug(
                f"Serving found image: {found_path} (root:{str(ranked_root)}) (parent: {found_path.parent})"
            )
            return send_file(str(found_path))
        except Exception as e:
            logger.exception(
                f"Failed to serve from parent dir, falling back to send_file: {e}"
            )
            return send_file(str(found_path))

    # Not found
    logger.warning(f"Image not found in ranked folders: {filepath}")
    return {"error": "Image not found"}, 404


@app.route("/images/<path:filename>")
def serve_image_by_name(filename: str):
    """Serve an image by filename. Looks up by provided score query, DB, or recursive search."""
    from file_management.path_handler import (
        get_ranked_root,
        compute_path_from_filename,
        find_image_path,
    )
    from database.images_table import get_image as get_db_image
    from pathlib import Path

    ranked_root = get_ranked_root()
    try:
        fname = Path(unquote(filename)).name
    except Exception:
        fname = Path(filename).name

    # Prefer explicit score query param when provided
    try:
        score_q = request.args.get("score", type=float)
    except Exception:
        score_q = None

    # 1) If score provided, compute the expected path and try to serve
    if score_q is not None:
        try:
            dest = compute_path_from_filename(fname, score_q)
            if dest.exists() and dest.is_file():
                return send_file(str(dest))
        except Exception:
            pass

    # 2) Try DB score
    try:
        db_entry = get_db_image(fname)
        if db_entry and db_entry.get("score") is not None:
            try:
                dest = compute_path_from_filename(fname, db_entry["score"])
                if dest.exists() and dest.is_file():
                    return send_file(str(dest))
            except Exception:
                pass
    except Exception:
        pass

    # 3) Fallback: recursive search
    try:
        found = find_image_path(fname)
        if found:
            return send_file(str(found))
    except Exception:
        pass

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
    try:
        return send_from_directory(str(base_dir / "frontend" / "html"), filename)
    except:
        return not_found(None)


# Error handlers
@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    return {"error": "Not found"}, 404


@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors."""
    print(f"Server error: {e}")
    return {"error": "Server error"}, 500


# Debug: Print all registered routes
print("[DEBUG] Registered routes:")
for rule in app.url_map.iter_rules():
    print(f"  {rule.rule} -> {rule.endpoint}")


def main():
    """Main entry point."""
    import argparse

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
        "--test",
        action="store_true",
        help="Run tests instead of starting server",
    )

    args = parser.parse_args()

    # Handle test mode
    if args.test:
        print("Running integration tests...")
        from tests import run_tests

        success = run_tests(verbose=True)
        return 0 if success else 1

    # Initialize system
    # If debug mode is ON, Flask runs twice (reloader). 
    # We only want to initialize in the actual worker process.
    should_init = True
    if args.debug and os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        should_init = False

    if should_init:
        if not init_ranking_system(args.image_root):
            print("ERROR: Failed to initialize system")
            return 1

    # Start server
    print(f"Starting ranking server on {args.host}:{args.port}...\n")
    # print("API Endpoints:")
    # print("  GET  /api/v2/ranking/status")
    # print("  GET  /api/v2/ranking/next-pair")
    # print("  POST /api/v2/ranking/submit-comparison")
    # print("  GET  /api/v2/gallery/images")
    # print("  GET  /api/v2/gallery/tier/<tier>")
    # print("  GET  /api/v2/gallery/search")
    # print("\nWeb UI: http://localhost:5001\n")

    try:
        app.run(host=args.host, port=args.port, debug=args.debug)
    except KeyboardInterrupt:
        print("\n\nServer stopped")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
