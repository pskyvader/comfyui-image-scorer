from prepare.data import manager
from prepare.config import schema as schema_module


def test_load_index_missing_returns_empty(tmp_path):
    path = tmp_path / "index.json"
    assert manager.load_index(str(path)) == []


def test_pad_existing_vectors_no_file(tmp_path):
    vectors_path = tmp_path / "vectors.json"
    manager.pad_existing_vectors(str(vectors_path), schema={})
    assert not vectors_path.exists()
