from __future__ import annotations

from .server.image_processor import ImageProcessor
import time

from ...shared.logger import get_logger, ModuleLogger
logger: ModuleLogger = get_logger(__name__)


def test_clean_json_metadata_initializes_neutral_fields(monkeypatch):
    _start = time.perf_counter()
    _start = time.perf_counter()
    _start = time.perf_counter()
    monkeypatch.setattr(
        ImageProcessor, "sync_processed_images_from_db", lambda self: None
    )
    processor = ImageProcessor(max_workers=1)
    cleaned = processor.clean_json_metadata(
        {
            "positive_prompt": "portrait, cinematic lighting",
            "confidence": 0.9,
            "score_modifier": 5,
        },
        default_score=0.5,
        filename="sample.png",
    )

    assert cleaned["score"] == 0.5
    assert cleaned["rating_mu"] > 0
    assert cleaned["rating_sigma"] > 0
    assert cleaned["comparison_count"] == 0
    assert cleaned["comparison_history"] == []
    assert "confidence" not in cleaned
    assert cleaned["filename"] == "sample.png"
