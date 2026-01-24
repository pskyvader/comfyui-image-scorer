from ranking.utils import find_images, load_metadata


def test_load_metadata_missing(tmp_path):
    img = tmp_path / "example.png"
    img.write_bytes(b"binary")
    assert load_metadata(str(img)) is None


def test_find_images_filters_extensions(tmp_path):
    (tmp_path / "a.png").write_bytes(b"")
    (tmp_path / "b.txt").write_text("noop", encoding="utf-8")
    files = find_images(str(tmp_path))
    assert files == [str(tmp_path / "a.png").replace('\\', '/')] 
