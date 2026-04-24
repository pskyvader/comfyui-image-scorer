"""Database module - SQLite schema and operations for ranking system."""

import sys
from pathlib import Path

# Set up path for shared imports
_root = Path(__file__).parent.parent.parent  # comfyui-image-scorer
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
