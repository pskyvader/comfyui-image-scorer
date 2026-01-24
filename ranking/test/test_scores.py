import sys
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ranking import scores

def test_normalize_items_dict():
    data = {"image_path": 5}
    items, err = scores._normalize_items(data)
    assert err is None
    assert len(items) == 1
    assert items[0]['image'] == "image_path"
    assert items[0]['score'] == 5

def test_normalize_items_list():
    data = [{"image": "p1", "score": 1}]
    items, err = scores._normalize_items(data)
    assert err is None
    assert len(items) == 1

def test_normalize_items_single_dict_with_image():
    data = {"image": "p1", "score": 1}
    items, err = scores._normalize_items(data)
    assert err is None
    assert len(items) == 1
    assert items[0]['image'] == "p1"

def test_normalize_items_invalid():
    with patch('ranking.scores.jsonify') as mock_json:
        mock_json.return_value = "error_resp"
        data = "string"
        items, err = scores._normalize_items(data)
        assert err is not None
        assert items == []
