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

from shared.analysis.mediapipe_analysis import MediaPipeAnalyzer


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
    assert analyzer._hand_landmarker is None


def test_analyze_returns_all_keys(analyzer: MediaPipeAnalyzer, blank_image: Image.Image) -> None:
    result = analyzer.analyze(blank_image)
    expected_keys = {"face_bbox", "body_pose", "left_hand", "right_hand"}
    assert result.keys() == expected_keys


def test_face_bbox_format(analyzer: MediaPipeAnalyzer, blank_image: Image.Image) -> None:
    result = analyzer.analyze(blank_image)
    for face in result["face_bbox"]:
        assert len(face) == 5
        x, y, w, h, conf = face
        assert 0.0 <= x <= 1.0
        assert 0.0 <= y <= 1.0
        assert 0.0 <= w <= 1.0
        assert 0.0 <= h <= 1.0
        assert 0.0 <= conf <= 1.0


def test_body_pose_landmark_count(analyzer: MediaPipeAnalyzer, noise_image: Image.Image) -> None:
    result = analyzer.analyze(noise_image)
    for person in result["body_pose"]:
        assert len(person) == 132


def test_hand_landmark_count(analyzer: MediaPipeAnalyzer, blank_image: Image.Image) -> None:
    result = analyzer.analyze(blank_image)
    for hand in result["left_hand"]:
        assert len(hand) == 63
    for hand in result["right_hand"]:
        assert len(hand) == 63


def test_lazy_loading_creates_models(analyzer: MediaPipeAnalyzer, blank_image: Image.Image) -> None:
    assert analyzer._face_detector is None
    try:
        analyzer.analyze(blank_image)
    except Exception:
        pass
    assert analyzer._face_detector is not None
    assert analyzer._pose_landmarker is not None
    assert analyzer._hand_landmarker is not None
