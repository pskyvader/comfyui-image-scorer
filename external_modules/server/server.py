"""Main server - Flask application for ranking system."""

import sys
import threading
import time
import os
from pathlib import Path
from flask import Flask, send_from_directory, request, send_file, Response
from urllib.parse import unquote
import argparse
import logging
from typing import Callable, ClassVar

# Set up paths FIRST before any imports
current_dir = str(Path(__file__).parent)
root_path = str(Path(__file__).parents[2])  # comfyui-image-scorer
custom_nodes_path = str(Path(__file__).parents[3])

if custom_nodes_path not in sys.path:
    sys.path.insert(0, custom_nodes_path)
if root_path not in sys.path:
    sys.path.insert(0, root_path)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if __name__ == "__main__":
    if __package__ is None:
        __package__ = "comfyui_image_scorer.external_modules.server"


# Now import Flask and API modules
from ..data_transform.endpoints import register_data_transform_routes

from ..comparison.endpoints import register_ranking_routes
from ..gallery.endpoints import register_gallery_routes
from ..maps.endpoints import register_maps_routes
from ..database_structure.endpoints import register_database_routes
from ..training_hyperparameters.endpoints import register_training_routes
from ..analysis.endpoints import register_analysis_routes
from ..database_structure.folder_organizer import ensure_tier_structure
from ..database_structure.path_handler import (
    get_ranked_root,
    compute_path_from_filename,
    find_image_path,
)
from ..database_structure.images_table import get_image as get_db_image
from .image_processor import ImageProcessor

from ...shared.config import config
from ...shared.paths import image_root
from ...shared.logger import (
    SSELogBroadcaster,
    SharedLogger,
    get_logger,
    set_log_filter_hook,
)

logger = get_logger(__name__)

# Global image processor
image_processor = ImageProcessor(max_workers=int(config["ranking"]["max_workers"]))

app = Flask(__name__, static_folder=None)
app.extensions["image_processor"] = image_processor
setattr(
    app, "image_processor", image_processor
)  # Also set as attribute for easy access

# logger: logging.Logger = app.logger


# werkzeug_logger = logging.getLogger("werkzeug")
# werkzeug_logger.setLevel(logging.WARNING)


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


# ── SSE log stream endpoint ──────────────────────────────────────────
@app.route("/api/v2/logs/stream")
def sse_log_stream():
    sub_id, q = SSELogBroadcaster.subscribe()

    def generate():
        try:
            yield f":connected\n\n"
            while True:
                line = q.get()
                yield f"data: {line}\n\n"
        except GeneratorExit:
            pass
        finally:
            SSELogBroadcaster.unsubscribe(sub_id)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# Global image processor state
scanner_thread: threading.Thread | None = None


def scanner_task(img_root: str) -> None:
    sleep_time = 30
    while True:
        _start = time.perf_counter()
        stats = image_processor.process_next_batch(img_root, batch_size=100)
        added = stats["added"]
        if added > 0:
            sleep_time = 30
        else:
            sleep_time *= 2
            sleep_time = min(sleep_time, 600)

        logger.info(f"Added:{added}, Sleeping {sleep_time}s...")
        time.sleep(sleep_time)


def start_background_scanner(img_root: str) -> None:
    """Start background thread to perform global image initialization."""
    _start = time.perf_counter()
    global scanner_thread
    if scanner_thread:

        logger.info(
            f"[SCANNER] Scanner already running. Alive: {scanner_thread.is_alive()}"
        )
        return
    scanner_thread = threading.Thread(
        target=scanner_task, daemon=True, args=(img_root,)
    )
    scanner_thread.start()

    logger.info("[SCANNER] Global image scanner started.")


def startup_worker(img_root: str, sync_existing: bool) -> None:
    """Background worker for initialization tasks."""
    _start = time.perf_counter()
    if not ensure_tier_structure():

        return

    if sync_existing:
        image_processor.rebuild_database_from_ranked()

    if img_root:
        scanner_task(img_root)


def init_ranking_system(img_root: str | None, sync_existing: bool) -> bool:
    """Initialize system and trigger background recovery."""
    _start = time.perf_counter()
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
def serve_index() -> Response:
    """Serve main index page."""
    _start = time.perf_counter()
    result = send_from_directory(str(SERVER_FRONTEND / "html"), "index.html")

    return result


@app.route("/css/<path:filename>")
def serve_css(filename: str) -> Response:
    """Serve CSS files (legacy - from server frontend)."""
    _start = time.perf_counter()
    result = send_from_directory(str(SERVER_FRONTEND / "css"), filename)

    return result


@app.route("/js/<path:filename>")
def serve_js(filename: str) -> Response:
    """Serve JavaScript files (legacy - from server frontend)."""
    _start = time.perf_counter()
    result = send_from_directory(str(SERVER_FRONTEND / "js"), filename)

    return result


@app.route("/static/<section>/<path:filename>")
def serve_section_static(section: str, filename: str):
    """Serve static files from a section's frontend/ folder."""
    _start = time.perf_counter()
    if section not in SECTION_FRONTENDS:
        result = {"error": f"Unknown section: {section}"}, 404

        return result
    result = send_from_directory(str(SECTION_FRONTENDS[section]), filename)

    return result


@app.route("/output/ranked/<path:filepath>")
def serve_ranked_image(filepath: str):
    """Serve images from scored folders."""
    _start = time.perf_counter()
    ranked_root = get_ranked_root()

    filepath_decoded = unquote(filepath)

    direct_path = ranked_root / filepath_decoded
    if direct_path.exists() and direct_path.is_file():
        response = send_file(str(direct_path))
        response.headers["Cache-Control"] = (
            "no-store, no-cache, must-revalidate, max-age=0"
        )
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"

        return response

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

        return response

    logger.warning(f"Image not found in ranked folders: {filepath}")
    result = {"error": "Image not found"}, 404

    return result


@app.route("/images/<path:filename>")
def serve_image_by_name(filename: str):
    """Serve an image by filename."""
    _start = time.perf_counter()
    ranked_root = get_ranked_root()
    fname = Path(unquote(filename)).name

    score_q = float(request.args["score"]) if "score" in request.args else None

    if score_q is not None:
        dest = compute_path_from_filename(fname, score_q)
        if dest.exists() and dest.is_file():

            return send_file(str(dest))

    db_entry = get_db_image(fname)
    if db_entry and db_entry["score"] is not None:
        dest = compute_path_from_filename(fname, db_entry["score"])
        if dest.exists() and dest.is_file():

            return send_file(str(dest))

    found = find_image_path(fname)
    if found:

        return send_file(str(found))

    return {"error": "Image not found"}, 404


@app.route("/image/<path:filename>")
def serve_image_alias(filename: str) -> Response:
    """Compatibility alias for legacy frontends that use `/image/`."""
    _start = time.perf_counter()
    result = serve_image_by_name(filename)

    return result


@app.route("/api/<path:path>")
def catch_api_404(path: str):
    """Explicitly handle unmapped API paths."""
    _start = time.perf_counter()
    result = {"error": f"API endpoint not found: /api/{path}"}, 404

    return result


@app.route("/<path:filename>")
def serve_html(filename: str) -> Response:
    """Serve HTML files - only matches files in frontend/html."""
    _start = time.perf_counter()
    base_dir = Path(__file__).parent
    result = send_from_directory(str(base_dir / "frontend" / "html"), filename)

    return result


@app.errorhandler(404)
def not_found(e: Exception):
    """Handle 404 errors."""
    _start = time.perf_counter()
    result = {"error": "Not found"}, 404

    return result


@app.errorhandler(500)
def server_error(e: Exception):
    """Handle 500 errors."""
    _start = time.perf_counter()
    logger.error(f"Server error: {e}")
    result = {"error": "Server error"}, 500

    return result


def set_log_filter(
    hook: Callable[[str, str | None], bool] | None = None,
    *,
    exact_names: set[str] | frozenset[str] | tuple[str, ...] | list[str] = (),
    prefixes: tuple[str, ...] | list[str] = (),
) -> None:
    """Configure a single filter that controls ALL output channels:

    - Console (via Python logging handlers)
    - Task buffer (via ``_TaskOutput``)
    - SSE stream (via ``SSELogBroadcaster``)

    The *hook* is called for **every** output line with
    ``(line_text, module_name)``.  Return ``True`` to allow the
    line or ``False`` to suppress it everywhere.

    Additionally *exact_names* / *prefixes* restrict which module
    names are allowed (same as ``SharedLogger.set_name_filters``).

    Examples::

        # Block messages containing "debug" in any channel.
        set_log_filter(
            hook=lambda line, mod: "debug" not in line,
        )

        # Block one specific function's output.
        set_log_filter(
            hook=lambda line, mod: mod != "shared.graph.noisy",
        )

        # Only allow ``shared.graph.crystal_graph``.
        set_log_filter(exact_names={"shared.graph.crystal_graph"})

        # Allow everything (default).
        set_log_filter()
    """
    set_log_filter_hook(hook)

    SharedLogger.set_name_filters(
        exact_names=exact_names,
        prefixes=prefixes,
    )

    class _ServerFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            if hook is not None and not hook(record.getMessage(), record.name):
                return False
            return SharedLogger.should_emit(record.name)

    for handler in logging.root.handlers:
        handler.addFilter(_ServerFilter())


def main() -> int:
    """Main entry point."""
    _start = time.perf_counter()

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

    log_level = logging.DEBUG if args.debug else logging.INFO

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
        # force=True,
    )

    # Suppress verbose third‑party loggers that have zero user value.
    logging.getLogger("PIL").setLevel(logging.WARNING)

    # ── Uncomment to activate module / content filtering ──────────────
    # set_log_filter(exact_names={"shared.graph.crystal_graph"})
    # set_log_filter(prefixes=("shared.",))
    # set_log_filter(hook=lambda line, mod: "accuracy" in line)

    should_init = True
    if args.debug and (
        "WERKZEUG_RUN_MAIN" not in os.environ
        or os.environ["WERKZEUG_RUN_MAIN"] != "true"
    ):
        should_init = False

    if should_init:
        if not init_ranking_system(args.image_root, sync_existing=args.sync_existing):
            return 1

    logger.info(f"Starting ranking server on {args.host}:{args.port}...\n")

    app.run(host=args.host, port=args.port, debug=args.debug)

    return 0


if __name__ == "__main__":
    sys.exit(main())
