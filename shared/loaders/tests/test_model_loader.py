from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

torch_stub = types.ModuleType("torch")
torch_stub.nn = types.SimpleNamespace(Module=object)  # type: ignore[attr-defined]
torch_stub.cuda = types.SimpleNamespace(
    get_device_properties=lambda device: types.SimpleNamespace(total_memory=123456789)
)
sys.modules.setdefault("torch", torch_stub)

timm_stub = types.ModuleType("timm")
timm_stub.create_model = lambda *args, **kwargs: None  # type: ignore[attr-defined]
sys.modules.setdefault("timm", timm_stub)

sentence_transformers_stub = types.ModuleType("sentence_transformers")
sentence_transformers_stub.SentenceTransformer = object  # type: ignore[attr-defined]
sys.modules.setdefault("sentence_transformers", sentence_transformers_stub)

shared_config_stub = types.ModuleType("shared.config")
shared_config_stub.config = {
    "prepare": {
        "vision_model": {"device": "cpu", "name": "vision", "output_dim": 64},
        "prompt_representation": {
            "device": "cpu",
            "name": "embedding",
            "output_dim": 128,
        },
    }
}
sys.modules.setdefault("shared.config", shared_config_stub)

from ...loaders import model_loader


def test_model_loader_init_uses_prepare_config() -> None:
    loader = model_loader.ModelLoader()
    assert loader.prepare_config is shared_config_stub.config["prepare"]


def test_load_vision_model_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.eval_called = False
            self.to_device = None

        def eval(self):
            self.eval_called = True
            return self

        def to(self, device):
            self.to_device = device
            return self

    fake_model = FakeModel()
    vision_config = model_loader.model_loader.prepare_config.setdefault("vision_models", {})
    vision_config["test_model"] = {"device": "cuda", "name": "fake", "output_dim": 256}
    monkeypatch.setattr(model_loader.timm, "create_model", lambda *args, **kwargs: fake_model)

    result = model_loader.model_loader.load_vision_model(model_key="test_model")
    cached = model_loader.model_loader.load_vision_model(model_key="test_model")

    assert result[0] is fake_model
    assert result[1] == 256
    assert result[2] == 123456789
    assert cached is result
    assert fake_model.eval_called is True
    assert fake_model.to_device == "cuda"


def test_load_embedding_model_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeSentenceTransformer:
        def __init__(self, name, device=None, local_files_only=None) -> None:
            self.name = name
            self.device = device
            self.local_files_only = local_files_only

    monkeypatch.setitem(
        model_loader.model_loader.prepare_config["prompt_representation"],
        "device",
        "cuda",
    )
    monkeypatch.setitem(
        model_loader.model_loader.prepare_config["prompt_representation"],
        "name",
        "fake-embed",
    )
    monkeypatch.setitem(
        model_loader.model_loader.prepare_config["prompt_representation"],
        "output_dim",
        384,
    )
    monkeypatch.setattr(model_loader, "SentenceTransformer", FakeSentenceTransformer)

    result = model_loader.model_loader.load_embedding_model()
    cached = model_loader.model_loader.load_embedding_model()

    assert result[0].name == "fake-embed"
    assert result[1] == 384
    assert cached is result
