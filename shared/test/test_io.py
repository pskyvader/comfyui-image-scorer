import json
from shared.io import atomic_write_json, load_index_set, load_json


def test_load_json_missing_returns_default(tmp_path):
    missing = tmp_path / "missing.json"
    data, err = load_json(str(missing), expect=dict, default={"fallback": 1})
    assert data == {"fallback": 1}
    assert err == "not_found"


def test_atomic_write_json_round_trip(tmp_path):
    target = tmp_path / "out.json"
    payload = {"a": 1, "b": [1, 2, 3]}
    atomic_write_json(str(target), payload, indent=None)
    with target.open("r", encoding="utf-8") as fh:
        loaded = json.load(fh)
    assert loaded == payload


def test_load_index_set_type_mismatch(tmp_path):
    bad_path = tmp_path / "bad.json"
    bad_path.write_text(json.dumps({"not": "a list"}), encoding="utf-8")
    data, err = load_index_set(str(bad_path), [])
    assert data == set()
    assert err == "invalid_type"
