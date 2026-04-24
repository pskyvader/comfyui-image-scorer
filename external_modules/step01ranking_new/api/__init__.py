"""API module - Flask routes and request handlers."""

import sys
from pathlib import Path

# Set up path for shared imports
_root = Path(__file__).parent.parent.parent  # comfyui-image-scorer
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from .ranking_api_v2 import register_ranking_routes
from .gallery_api import register_gallery_routes

__all__ = [
    "register_ranking_routes",
    "register_gallery_routes",
]
