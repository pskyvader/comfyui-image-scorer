from typing import Any

try:
    from .nodes.aesthetic_score.node import AestheticScoreNode
except ImportError:
    # Handle during test collection when relative imports aren't available
    AestheticScoreNode = None

# from .text_score.node import TextScoreNode

NODE_CLASS_MAPPINGS: dict[str, Any] = {
    "AestheticScoreNode": AestheticScoreNode,
    # "TextScoreNode": TextScoreNode,
} if AestheticScoreNode else {}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "AestheticScoreNode": "Calculate Aesthetic Score",
    # "TextScoreNode": "Score Text+Params",
} if AestheticScoreNode else {}

__all__: list[str] = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
