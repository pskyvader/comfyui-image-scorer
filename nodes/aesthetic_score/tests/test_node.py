from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _install_node_stubs() -> None:
    torch_stub = types.ModuleType("torch")
    torch_stub.Tensor = object  # type: ignore[attr-defined]
    sys.modules.setdefault("torch", torch_stub)

    helpers_stub = types.ModuleType("shared.helpers")
    helpers_stub.export_image_batch = lambda images: list(images)  # type: ignore[attr-defined]
    sys.modules.setdefault("shared.helpers", helpers_stub)

    vectors_stub = types.ModuleType("shared.vectors.vectors")

    class _StubVectorList:
        def __init__(
            self,
            raw_data,
            index_list,
            vectors_list,
            scores_list,
            text_list,
            add_new,
            merge_lists=False,
            read_only=False,
            process_images=True,
        ) -> None:
            self.raw_data = raw_data
            self.sorted_vectors = {
                "image": {
                    "vector": types.SimpleNamespace(vector_list=[]),
                    "name": "image",
                    "slot_size": 1,
                    "type": "image",
                }
            }

        def create_vectors(self) -> None:
            return None

        def join_vectors(self):
            image_vector = self.sorted_vectors["image"]["vector"]
            if not image_vector.vector_list:
                image_vector.vector_list = [[float(index + 1)] for index in range(len(self.raw_data))]
            return image_vector.vector_list

    vectors_stub.VectorList = _StubVectorList  # type: ignore[attr-defined]
    sys.modules.setdefault("shared.vectors.vectors", vectors_stub)

    image_analysis_stub = types.ModuleType("shared.image_analysis")

    class _StubImageAnalysis:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def prepare_image_batch(self, image):
            return list(image)

        def analyze_image_batch(self, current_batch, data_batch):
            return data_batch

    image_analysis_stub.ImageAnalysis = _StubImageAnalysis  # type: ignore[attr-defined]
    sys.modules.setdefault("shared.image_analysis", image_analysis_stub)

    image_vector_stub = types.ModuleType("shared.vectors.image_vector")

    class _StubImageVector:
        def __init__(self, name: str) -> None:
            self.name = name
            self.image_list = []
            self.vector_list = []

        def create_vector_list(self, memory_usage: float = 0.85, rebuild: bool = False):
            self.vector_list = [[float(index + 1)] for index, _ in enumerate(self.image_list)]
            return self.vector_list

    image_vector_stub.ImageVector = _StubImageVector  # type: ignore[attr-defined]
    sys.modules.setdefault("shared.vectors.image_vector", image_vector_stub)

    training_package_stub = types.ModuleType("shared.training")
    training_package_stub.__path__ = [str(ROOT / "shared" / "training")]
    sys.modules.setdefault("shared.training", training_package_stub)

    data_transformer_stub = types.ModuleType("shared.training.data_transformer")
    data_transformer_stub.data_transformer = types.SimpleNamespace(
        apply_feature_filter=lambda vecs: vecs
    )
    sys.modules.setdefault("shared.training.data_transformer", data_transformer_stub)

    training_loader_stub = types.ModuleType("shared.loaders.training_loader")
    training_loader_stub.training_loader = types.SimpleNamespace(
        load_training_model=lambda: types.SimpleNamespace(
            predict=lambda features: np.array([0.2, 0.8, 0.5], dtype=np.float32)
        )
    )
    sys.modules.setdefault("shared.loaders.training_loader", training_loader_stub)

    project_pkg = types.ModuleType("comfyui_image_scorer")
    project_pkg.__path__ = [str(ROOT)]
    nodes_pkg = types.ModuleType("comfyui_image_scorer.nodes")
    nodes_pkg.__path__ = [str(ROOT / "nodes")]
    aesthetic_pkg = types.ModuleType("comfyui_image_scorer.nodes.aesthetic_score")
    aesthetic_pkg.__path__ = [str(ROOT / "nodes" / "aesthetic_score")]
    shared_pkg = types.ModuleType("comfyui_image_scorer.shared")
    shared_pkg.__path__ = [str(ROOT / "shared")]
    shared_vectors_pkg = types.ModuleType("comfyui_image_scorer.shared.vectors")
    shared_vectors_pkg.__path__ = [str(ROOT / "shared" / "vectors")]
    shared_loaders_pkg = types.ModuleType("comfyui_image_scorer.shared.loaders")
    shared_loaders_pkg.__path__ = [str(ROOT / "shared" / "loaders")]
    shared_training_pkg = types.ModuleType("comfyui_image_scorer.shared.training")
    shared_training_pkg.__path__ = [str(ROOT / "shared" / "training")]

    sys.modules.setdefault("comfyui_image_scorer", project_pkg)
    sys.modules.setdefault("comfyui_image_scorer.nodes", nodes_pkg)
    sys.modules.setdefault("comfyui_image_scorer.nodes.aesthetic_score", aesthetic_pkg)
    sys.modules.setdefault("comfyui_image_scorer.shared", shared_pkg)
    sys.modules.setdefault("comfyui_image_scorer.shared.vectors", shared_vectors_pkg)
    sys.modules.setdefault("comfyui_image_scorer.shared.loaders", shared_loaders_pkg)
    sys.modules.setdefault("comfyui_image_scorer.shared.training", shared_training_pkg)

    sys.modules["comfyui_image_scorer.shared.helpers"] = helpers_stub
    sys.modules["comfyui_image_scorer.shared.vectors.vectors"] = vectors_stub
    sys.modules["comfyui_image_scorer.shared.image_analysis"] = image_analysis_stub
    sys.modules["comfyui_image_scorer.shared.vectors.image_vector"] = image_vector_stub
    sys.modules["comfyui_image_scorer.shared.training.data_transformer"] = data_transformer_stub
    sys.modules["comfyui_image_scorer.shared.loaders.training_loader"] = training_loader_stub


_install_node_stubs()

node_path = ROOT / "nodes" / "aesthetic_score" / "node.py"
node_spec = importlib.util.spec_from_file_location(
    "comfyui_image_scorer.nodes.aesthetic_score.node",
    node_path,
)
assert node_spec and node_spec.loader
node_module = importlib.util.module_from_spec(node_spec)
sys.modules[node_spec.name] = node_module
node_spec.loader.exec_module(node_module)
AestheticScoreNode = node_module.AestheticScoreNode


def test_input_types_schema() -> None:
    schema = AestheticScoreNode.INPUT_TYPES()
    required = schema["required"]

    assert "image" in required
    assert "threshold" in required
    assert "positive" in required
    assert "negative" in required


def test_calculate_score_validation_and_selection() -> None:
    node = AestheticScoreNode()

    with pytest.raises(ValueError):
        node.calculate_score(
            image=[1],
            threshold=2.5,
            positive="",
            negative="bad",
            steps=20,
            cfg=7.0,
            sampler="euler",
            scheduler="normal",
            model_name="model",
            lora_name="lora",
            lora_strength=0.0,
        )

    selected, discarded, available, scores = node.calculate_score(
        image=[1, 2, 3],
        threshold=0.4,
        positive="good",
        negative="bad",
        steps=20,
        cfg=7.0,
        sampler="euler",
        scheduler="normal",
        model_name="model",
        lora_name="lora",
        lora_strength=0.0,
        min_images=1,
        max_images=2,
    )

    assert selected == [2, 3]
    assert discarded == [1]
    assert available is True
    assert len(scores) == 3
