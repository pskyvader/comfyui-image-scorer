from typing import Any, Dict
from .aesthetic_score.node import AestheticScoreNode
# from .text_score.node import TextScoreNode


NODE_CLASS_MAPPINGS: Dict[str, Any] = {
    "AestheticScoreNode": AestheticScoreNode,
    # "TextScoreNode": TextScoreNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AestheticScoreNode": "Calculate Aesthetic Score",
    # "TextScoreNode": "Score Text+Params",
}


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
