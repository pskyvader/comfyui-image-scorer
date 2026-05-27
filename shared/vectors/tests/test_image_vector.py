import pytest
import numpy as np
import torch
from PIL import Image
from unittest.mock import patch, MagicMock
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from shared.vectors.image_vector import (
    ImageConverter,
    VectorEncoder,
    PathProcessor,
    ImageVector,
)
from shared.vectors.batch_sizer import BatchSizer, HistoryEntry, ProfileData
import logging
import time
logger = logging.getLogger(__name__)


def test_image_converter_to_pil_3d_rgb():
    _start = time.perf_counter()
    _start = time.perf_counter()
    _start = time.perf_counter()
    arr = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    imgs = ImageConverter.to_pil(arr)
    assert len(imgs) == 1
    assert imgs[0].mode == "RGB"
    logger.debug("test_image_converter_to_pil_3d_rgb took %.4fs", time.perf_counter() - _start)


def test_image_converter_to_pil_2d():
    _start = time.perf_counter()
    _start = time.perf_counter()
    _start = time.perf_counter()
    arr = np.random.randint(0, 255, (32, 32), dtype=np.uint8)
    imgs = ImageConverter.to_pil(arr)
    assert len(imgs) == 1
    assert imgs[0].mode == "RGB"
    logger.debug("test_image_converter_to_pil_2d took %.4fs", time.perf_counter() - _start)


def test_image_converter_to_pil_chw():
    _start = time.perf_counter()
    _start = time.perf_counter()
    _start = time.perf_counter()
    arr = np.random.randint(0, 255, (3, 32, 32), dtype=np.uint8)
    imgs = ImageConverter.to_pil(arr)
    assert len(imgs) == 1
    logger.debug("test_image_converter_to_pil_chw took %.4fs", time.perf_counter() - _start)


def test_image_converter_to_pil_4d():
    _start = time.perf_counter()
    _start = time.perf_counter()
    _start = time.perf_counter()
    arr = np.random.randint(0, 255, (2, 32, 32, 3), dtype=np.uint8)
    imgs = ImageConverter.to_pil(arr)
    assert len(imgs) == 2
    assert all(isinstance(img, Image.Image) for img in imgs)
    logger.debug("test_image_converter_to_pil_4d took %.4fs", time.perf_counter() - _start)


def test_image_converter_from_path_success():
    _start = time.perf_counter()
    _start = time.perf_counter()
    _start = time.perf_counter()
    mock_img = MagicMock()
    mock_img.convert.return_value = mock_img
    with patch("PIL.Image.open", return_value=mock_img):
        result = ImageConverter.from_path("test.jpg")
        assert len(result) == 1
        logger.debug("test_image_converter_from_path_success took %.4fs", time.perf_counter() - _start)


def test_image_converter_from_path_failure():
    _start = time.perf_counter()
    _start = time.perf_counter()
    _start = time.perf_counter()
    with patch("PIL.Image.open", side_effect=Exception("fail")):
        result = ImageConverter.from_path("nonexistent.jpg")
        assert len(result) == 1
        assert isinstance(result[0], Image.Image)
        logger.debug("test_image_converter_from_path_failure took %.4fs", time.perf_counter() - _start)


def test_image_converter_prepare_string():
    _start = time.perf_counter()
    _start = time.perf_counter()
    _start = time.perf_counter()
    mock_img = MagicMock()
    mock_img.convert.return_value = mock_img
    with patch("PIL.Image.open", return_value=mock_img):
        result = ImageConverter.prepare("test.jpg")
        assert len(result) == 1
        logger.debug("test_image_converter_prepare_string took %.4fs", time.perf_counter() - _start)


def test_image_converter_prepare_pil():
    _start = time.perf_counter()
    _start = time.perf_counter()
    _start = time.perf_counter()
    img = Image.new("RGB", (32, 32))
    result = ImageConverter.prepare(img)
    assert len(result) == 1
    logger.debug("test_image_converter_prepare_pil took %.4fs", time.perf_counter() - _start)


def test_image_converter_prepare_tensor():
    _start = time.perf_counter()
    _start = time.perf_counter()
    _start = time.perf_counter()
    tensor = torch.rand(1, 32, 32, 3)
    result = ImageConverter.prepare(tensor)
    assert len(result) == 1
    logger.debug("test_image_converter_prepare_tensor took %.4fs", time.perf_counter() - _start)


def test_image_converter_prepare_ndarray():
    _start = time.perf_counter()
    _start = time.perf_counter()
    _start = time.perf_counter()
    arr = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    result = ImageConverter.prepare(arr)
    assert len(result) == 1
    logger.debug("test_image_converter_prepare_ndarray took %.4fs", time.perf_counter() - _start)


def test_image_converter_prepare_list():
    _start = time.perf_counter()
    _start = time.perf_counter()
    _start = time.perf_counter()
    arr = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    result = ImageConverter.prepare([arr])
    assert len(result) == 1
    logger.debug("test_image_converter_prepare_list took %.4fs", time.perf_counter() - _start)


def test_image_converter_prepare_unsupported():
    _start = time.perf_counter()
    _start = time.perf_counter()
    _start = time.perf_counter()
    with pytest.raises(TypeError):
        ImageConverter.prepare(123)
        logger.debug("test_image_converter_prepare_unsupported took %.4fs", time.perf_counter() - _start)


def test_image_converter_get_size_from_path():
    _start = time.perf_counter()
    _start = time.perf_counter()
    _start = time.perf_counter()
    mock_img = MagicMock()
    mock_img.size = (64, 64)
    mock_img.__enter__ = MagicMock(return_value=mock_img)
    mock_img.__exit__ = MagicMock(return_value=False)
    with patch("PIL.Image.open", return_value=mock_img):
        result = ImageConverter.get_size_from_path("test.jpg")
        assert result == (64, 64)
        logger.debug("test_image_converter_get_size_from_path took %.4fs", time.perf_counter() - _start)


def test_image_converter_get_size_from_path_failure():
    _start = time.perf_counter()
    _start = time.perf_counter()
    _start = time.perf_counter()
    with patch("PIL.Image.open", side_effect=Exception("fail")):
        result = ImageConverter.get_size_from_path("nonexistent.jpg")
        assert result == (512, 512)
        logger.debug("test_image_converter_get_size_from_path_failure took %.4fs", time.perf_counter() - _start)


def test_vector_encoder_get_transform():
    _start = time.perf_counter()
    _start = time.perf_counter()
    _start = time.perf_counter()
    VectorEncoder._transform = None
    transform = VectorEncoder.get_transform()
    assert transform is not None
    assert hasattr(transform, "__call__")
    cached = VectorEncoder.get_transform()
    assert cached is transform
    logger.debug("test_vector_encoder_get_transform took %.4fs", time.perf_counter() - _start)


def test_vector_encoder_encode():
    _start = time.perf_counter()
    _start = time.perf_counter()
    _start = time.perf_counter()
    mock_model = MagicMock()
    mock_model.parameters.return_value = iter([torch.nn.Parameter(torch.zeros(1))])
    mock_model.return_value = torch.rand(2, 512)
    images = [Image.new("RGB", (32, 32)) for _ in range(2)]
    transform = VectorEncoder.get_transform()
    result = VectorEncoder.encode(images, mock_model, 512, transform)
    assert isinstance(result, list)
    assert len(result) == 2
    logger.debug("test_vector_encoder_encode took %.4fs", time.perf_counter() - _start)


def test_vector_encoder_run_test():
    _start = time.perf_counter()
    _start = time.perf_counter()
    _start = time.perf_counter()
    mock_model = MagicMock()
    mock_model.parameters.return_value = [torch.nn.Parameter(torch.zeros(1))]
    batch_tensor = torch.rand(2, 3, 32, 32)
    with patch("torch.cuda.synchronize"):
        VectorEncoder.run_test(mock_model, batch_tensor, "cuda")
    mock_model.eval.assert_called_once()
    logger.debug("test_vector_encoder_run_test took %.4fs", time.perf_counter() - _start)


def test_batchSizer_init():
    _start = time.perf_counter()
    _start = time.perf_counter()
    _start = time.perf_counter()
    sizer = BatchSizer()
    assert sizer._active is None
    assert sizer._ready is False
    logger.debug("test_batchSizer_init took %.4fs", time.perf_counter() - _start)


def test_batchSizer_fit_model_insufficient_data():
    _start = time.perf_counter()
    _start = time.perf_counter()
    _start = time.perf_counter()
    sizer = BatchSizer()
    sizer._active = ProfileData(
        model_name="test", device_name="test", device_id="cuda:0",
        total_memory=8000000000, model_memory_bytes=1000000000,
    )
    sizer._active.history["224x224"] = [
        HistoryEntry(batch_size=1, delta_memory=500000000, timestamp=100.0),
    ]
    sizer._fit_model()
    assert sizer._active.pixel_cost is None
    assert sizer._active.fixed_overhead is None
    assert sizer._active.r_squared is None
    logger.debug("test_batchSizer_fit_model_insufficient_data took %.4fs", time.perf_counter() - _start)


def test_batchSizer_fit_model_two_entries():
    _start = time.perf_counter()
    _start = time.perf_counter()
    _start = time.perf_counter()
    sizer = BatchSizer()
    sizer._active = ProfileData(
        model_name="test", device_name="test", device_id="cuda:0",
        total_memory=8000000000, model_memory_bytes=1000000000,
    )
    sizer._active.history["224x224"] = [
        HistoryEntry(batch_size=1, delta_memory=500000000, timestamp=100.0),
        HistoryEntry(batch_size=4, delta_memory=800000000, timestamp=101.0),
    ]
    sizer._fit_model()
    assert sizer._active.pixel_cost is not None
    assert sizer._active.fixed_overhead is not None
    assert sizer._active.r_squared is not None
    assert sizer._active.pixel_cost > 0
    logger.debug("test_batchSizer_fit_model_two_entries took %.4fs", time.perf_counter() - _start)


def test_batchSizer_cache_hit():
    _start = time.perf_counter()
    _start = time.perf_counter()
    _start = time.perf_counter()
    sizer = BatchSizer()
    sizer._ready = True
    sizer._active = ProfileData(
        model_name="test", device_name="test", device_id="cuda:0",
        total_memory=8000000000, model_memory_bytes=1000000000,
    )
    sizer._active.history["64x64"] = [
        HistoryEntry(batch_size=10, delta_memory=200000000, timestamp=100.0),
    ]
    result = sizer.get(64, 64, rebuild=False)
    assert result == 10
    logger.debug("test_batchSizer_cache_hit took %.4fs", time.perf_counter() - _start)


def test_batchSizer_get_with_rebuild_no_history():
    _start = time.perf_counter()
    _start = time.perf_counter()
    _start = time.perf_counter()
    sizer = BatchSizer()
    sizer._ready = True
    sizer._active = ProfileData(
        model_name="test", device_name="test", device_id="cuda:0",
        total_memory=8000000000, model_memory_bytes=1000000000,
    )
    with patch.object(sizer, "_profile_new_resolution", return_value=5) as mock_profile:
        result = sizer.get(64, 64, rebuild=True)
        assert result == 5
        mock_profile.assert_called_once_with(64, 64, True)
        logger.debug("test_batchSizer_get_with_rebuild_no_history took %.4fs", time.perf_counter() - _start)


def test_path_processor_build_buckets():
    _start = time.perf_counter()
    _start = time.perf_counter()
    _start = time.perf_counter()
    path_list = ["img1.jpg", "img2.jpg", "img3.jpg"]
    with patch("os.path.exists", return_value=True), \
         patch("shared.vectors.image_vector.ImageConverter.get_size_from_path", return_value=(64, 64)):
        buckets = PathProcessor.build_buckets(path_list)
        assert len(buckets) == 1
        assert len(buckets[(64, 64)]) == 3
        logger.debug("test_path_processor_build_buckets took %.4fs", time.perf_counter() - _start)


def test_path_processor_sort_buckets():
    _start = time.perf_counter()
    _start = time.perf_counter()
    _start = time.perf_counter()
    buckets: dict[tuple[int, int], list] = {(64, 64): [(0, "a")], (128, 128): [(1, "b")]}
    result = PathProcessor.sort_buckets(buckets)
    keys = list(result.keys())
    assert keys[0] == (128, 128)
    logger.debug("test_path_processor_sort_buckets took %.4fs", time.perf_counter() - _start)


def test_path_processor_process_bucket_success():
    _start = time.perf_counter()
    _start = time.perf_counter()
    _start = time.perf_counter()
    mock_model = MagicMock()
    mock_model.parameters.return_value = iter([torch.nn.Parameter(torch.zeros(1))])
    mock_model.return_value = torch.rand(1, 512)
    items = [(0, "img1.jpg"), (1, "img2.jpg")]
    vectors: list[list[float]] = [[], []]
    mock_pbar = MagicMock()
    transform = VectorEncoder.get_transform()

    mock_batch_sizer = MagicMock(spec=BatchSizer)
    mock_batch_sizer.get.return_value = 10

    with patch("os.path.exists", return_value=True), \
         patch("shared.vectors.image_vector.ImageConverter.prepare", return_value=[Image.new("RGB", (32, 32))]), \
         patch("shared.vectors.image_vector.VectorEncoder.encode", return_value=[[0.1] * 512, [0.2] * 512]), \
         patch("torch.cuda.empty_cache"):
        result = PathProcessor.process_bucket(
            items, (32, 32), mock_model, 512, transform, mock_batch_sizer, 0.85, vectors, mock_pbar
        )
        assert result is None
        logger.debug("test_path_processor_process_bucket_success took %.4fs", time.perf_counter() - _start)


def test_path_processor_process_bucket_retry_rebuild():
    _start = time.perf_counter()
    _start = time.perf_counter()
    _start = time.perf_counter()
    mock_model = MagicMock()
    mock_model.parameters.return_value = iter([torch.nn.Parameter(torch.zeros(1))])
    mock_model.return_value = torch.rand(1, 512)
    items = [(0, "img1.jpg")]
    vectors: list[list[float]] = [[]]
    mock_pbar = MagicMock()
    transform = VectorEncoder.get_transform()
    call_count = {"count": 0}

        _start = time.perf_counter()
    def encode_side_effect(*args, **kwargs):
        _start = time.perf_counter()
        _start = time.perf_counter()
        call_count["count"] += 1
        if call_count["count"] == 1:
            raise RuntimeError("OOM")
        result = [[0.1] * 512]
        logger.debug("encode_side_effect took %.4fs", time.perf_counter() - _start)
        result = result
        logger.debug("test_path_processor_process_bucket_retry_rebuild took %.4fs", time.perf_counter() - _start)
        result = result
        logger.debug("encode_side_effect took %.4fs", time.perf_counter() - _start)
        result = 
        logger.debug("encode_side_effect took %.4fs", time.perf_counter() - _start)
        return result
        logger.debug("test_path_processor_process_bucket_retry_rebuild took %.4fs", time.perf_counter() - _start)
        return result

    mock_batch_sizer = MagicMock(spec=BatchSizer)
    mock_batch_sizer.get.side_effect = [10, 5]

    with patch("os.path.exists", return_value=True), \
         patch("shared.vectors.image_vector.ImageConverter.prepare", return_value=[Image.new("RGB", (32, 32))]), \
         patch("shared.vectors.image_vector.VectorEncoder.encode", side_effect=encode_side_effect), \
         patch("torch.cuda.empty_cache"):
        result = PathProcessor.process_bucket(
            items, (32, 32), mock_model, 512, transform, mock_batch_sizer, 0.85, vectors, mock_pbar
        )
        assert result is None
        assert mock_batch_sizer.get.call_count == 2
        assert mock_batch_sizer.get.call_args_list[0][0][2] is False
        assert mock_batch_sizer.get.call_args_list[1][0][2] is True
        logger.debug("test_path_processor_process_bucket_retry_rebuild took %.4fs", time.perf_counter() - _start)


def test_path_processor_process_bucket_fail_after_rebuild():
    _start = time.perf_counter()
    _start = time.perf_counter()
    _start = time.perf_counter()
    mock_model = MagicMock()
    mock_model.parameters.return_value = iter([torch.nn.Parameter(torch.zeros(1))])
    items = [(0, "img1.jpg")]
    vectors: list[list[float]] = [[]]
    mock_pbar = MagicMock()
    transform = VectorEncoder.get_transform()

    mock_batch_sizer = MagicMock(spec=BatchSizer)
    mock_batch_sizer.get.side_effect = [10, 5]

    with patch("os.path.exists", return_value=True), \
         patch("shared.vectors.image_vector.ImageConverter.prepare", return_value=[Image.new("RGB", (32, 32))]), \
         patch("shared.vectors.image_vector.VectorEncoder.encode", side_effect=RuntimeError("OOM")), \
         patch("torch.cuda.empty_cache"):
        result = PathProcessor.process_bucket(
            items, (32, 32), mock_model, 512, transform, mock_batch_sizer, 0.85, vectors, mock_pbar
        )
        assert result == (32, 32)
        logger.debug("test_path_processor_process_bucket_fail_after_rebuild took %.4fs", time.perf_counter() - _start)


def test_image_vector_init():
    _start = time.perf_counter()
    _start = time.perf_counter()
    _start = time.perf_counter()
    with patch("torch.cuda.set_per_process_memory_fraction"):
        iv = ImageVector("test")
        assert iv.name == "test"
        assert iv.image_list == []
        assert iv.path_list == []
        logger.debug("test_image_vector_init took %.4fs", time.perf_counter() - _start)


def test_image_vector_array_to_pil():
    _start = time.perf_counter()
    _start = time.perf_counter()
    _start = time.perf_counter()
    with patch("torch.cuda.set_per_process_memory_fraction"):
        iv = ImageVector("test")
        arr = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        result = iv.array_to_pil(arr)
        assert len(result) == 1
        logger.debug("test_image_vector_array_to_pil took %.4fs", time.perf_counter() - _start)


def test_image_vector_prepare_image_batch():
    _start = time.perf_counter()
    _start = time.perf_counter()
    _start = time.perf_counter()
    with patch("torch.cuda.set_per_process_memory_fraction"):
        iv = ImageVector("test")
        img = Image.new("RGB", (32, 32))
        result = iv.prepare_image_batch(img)
        assert len(result) == 1
        logger.debug("test_image_vector_prepare_image_batch took %.4fs", time.perf_counter() - _start)


def test_image_vector_get_batch_size():
    _start = time.perf_counter()
    _start = time.perf_counter()
    _start = time.perf_counter()
    mock_batch_sizer = MagicMock(spec=BatchSizer)
    mock_batch_sizer.get.return_value = 10

    with patch("torch.cuda.set_per_process_memory_fraction"), \
         patch("shared.vectors.image_vector.BatchSizer", return_value=mock_batch_sizer):
        iv = ImageVector("test")
        result = iv.get_batch_size(64, 64, rebuild=False)
        assert result == 10
        mock_batch_sizer.get.assert_called_once_with(64, 64, False)
        logger.debug("test_image_vector_get_batch_size took %.4fs", time.perf_counter() - _start)


def test_image_vector_create_vector_list_empty():
    _start = time.perf_counter()
    _start = time.perf_counter()
    _start = time.perf_counter()
    with patch("torch.cuda.set_per_process_memory_fraction"):
        iv = ImageVector("test")
        result = iv.create_vector_list()
        assert result == []
        logger.debug("test_image_vector_create_vector_list_empty took %.4fs", time.perf_counter() - _start)


def test_image_vector_create_vector_list():
    _start = time.perf_counter()
    _start = time.perf_counter()
    _start = time.perf_counter()
    mock_model = MagicMock()
    mock_model.parameters.return_value = iter([torch.nn.Parameter(torch.zeros(1))])
    mock_model.return_value = torch.rand(1, 512)

    mock_batch_sizer = MagicMock(spec=BatchSizer)
    mock_batch_sizer.get.return_value = 10

    with patch("torch.cuda.set_per_process_memory_fraction"), \
         patch("shared.vectors.image_vector.model_loader") as mock_loader, \
         patch("shared.vectors.image_vector.BatchSizer", return_value=mock_batch_sizer), \
         patch("shared.vectors.image_vector.VectorEncoder.encode", return_value=[[0.1] * 512]):
        mock_loader.load_vision_model.return_value = (mock_model, 512, 8000000000)
        iv = ImageVector("test")
        iv.image_list = [Image.new("RGB", (32, 32))]
        result = iv.create_vector_list()
        assert isinstance(result, list)
        logger.debug("test_image_vector_create_vector_list took %.4fs", time.perf_counter() - _start)


def test_image_vector_create_vector_list_from_paths_empty():
    _start = time.perf_counter()
    _start = time.perf_counter()
    _start = time.perf_counter()
    with patch("torch.cuda.set_per_process_memory_fraction"):
        iv = ImageVector("test")
        result = iv.create_vector_list_from_paths()
        assert result == []
        logger.debug("test_image_vector_create_vector_list_from_paths_empty took %.4fs", time.perf_counter() - _start)
