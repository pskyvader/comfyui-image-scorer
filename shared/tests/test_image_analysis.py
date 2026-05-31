from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.image_analysis import ImageAnalysis, process_single_batch


def _make_image(path: Path) -> None:
    grid = np.indices((128, 128)).sum(axis=0).astype(np.uint8)
    rgb = np.stack([grid, np.flipud(grid), np.fliplr(grid)], axis=-1)
    Image.fromarray(rgb, mode="RGB").save(path)


def test_process_single_batch_passthrough(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    _make_image(image_path)

    data = [(str(image_path), {"score": 1, "width": 128, "height": 128, "aspect_ratio": 1.0}, "ts", "id")]

    def prepare(paths: list[str]) -> list[Image.Image]:
        assert paths == [str(image_path)]
        return [Image.open(paths[0]).convert("RGB")]

    def analyze(
        image_batch: list[Image.Image],
        batch_data: list[tuple[str, dict[str, object], str, str]],
    ) -> list[tuple[str, dict[str, object], str, str]]:
        assert len(image_batch) == 1
        batch_data[0][1]["flag"] = True
        return batch_data

    result = process_single_batch(prepare, analyze, [str(image_path)], data)
    assert result[0][1]["flag"] is True


def test_analyze_images_from_paths(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    _make_image(image_path)

    raw_data = [
        (
            str(image_path),
            {
                "score": 1.0,
                "width": 128,
                "height": 128,
                "aspect_ratio": 1.0,
            },
            "ts",
            "id",
        )
    ]

    analysis = ImageAnalysis(raw_data)
    result = analysis.analyze_images_from_paths(batch_size=1, max_workers=1)

    assert len(result) == 1
    entry = result[0][1]
    assert entry["original_width"] == 128
    assert entry["original_height"] == 128
    assert entry["final_width"] == 128
    assert entry["final_height"] == 128
    assert "contrast" in entry
    assert "sharpness" in entry
    assert "noise_score" in entry
    assert "colorfulness" in entry
    assert "artifact_score" in entry
    assert "edge_density" in entry
    assert "texture_lbp" in entry
    assert analysis.processed_data == result
