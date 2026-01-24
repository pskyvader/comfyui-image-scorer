import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training import helpers



def test_resolve_path_relative():
    rel_path = "training/output"
    resolved = helpers.resolve_path(rel_path)
    assert Path(resolved).is_absolute()
    assert str(Path(resolved)).endswith(rel_path.replace("/", "\\")) or str(Path(resolved)).endswith(rel_path)
