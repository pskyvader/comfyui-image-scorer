from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.analysis.mediapipe_analysis import (
    MediaPipeAnalyzer,
    POSE_LANDMARK_NAMES,
)


@pytest.fixture
def analyzer() -> MediaPipeAnalyzer:
    return MediaPipeAnalyzer()


@pytest.fixture
def blank_image() -> Image.Image:
    return Image.new("RGB", (224, 224), (128, 128, 128))


@pytest.fixture
def noise_image() -> Image.Image:
    arr = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def test_analyzer_initializes(analyzer: MediaPipeAnalyzer) -> None:
    assert analyzer._face_detector is None
    assert analyzer._pose_landmarker is None


def test_analyze_returns_all_keys(analyzer: MediaPipeAnalyzer, blank_image: Image.Image) -> None:
    result = analyzer.analyze(blank_image)
    expected_keys = {"bbox"} | set(POSE_LANDMARK_NAMES)
    assert result.keys() == expected_keys


def test_face_bbox_format(analyzer: MediaPipeAnalyzer, blank_image: Image.Image) -> None:
    result = analyzer.analyze(blank_image)
    for face in result["bbox"]:
        assert set(face.keys()) == {"x", "y", "width", "height", "confidence"}
        assert 0.0 <= face["x"] <= 1.0
        assert 0.0 <= face["y"] <= 1.0
        assert 0.0 <= face["width"] <= 1.0
        assert 0.0 <= face["height"] <= 1.0
        assert 0.0 <= face["confidence"] <= 1.0


def test_body_pose_landmark_count(analyzer: MediaPipeAnalyzer, noise_image: Image.Image) -> None:
    result = analyzer.analyze(noise_image)
    for name in POSE_LANDMARK_NAMES:
        assert name in result
        for person in result[name]:
            assert set(person.keys()) == {"x", "y", "z", "visibility"}


def test_lazy_loading_creates_models(analyzer: MediaPipeAnalyzer, blank_image: Image.Image) -> None:
    assert analyzer._face_detector is None
    try:
        analyzer.analyze(blank_image)
    except Exception:
        pass
    assert analyzer._face_detector is not None
    assert analyzer._pose_landmarker is not None
