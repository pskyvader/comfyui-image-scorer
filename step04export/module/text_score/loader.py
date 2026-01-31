

class TextScorerLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("TEXT_SCORER",)
    RETURN_NAMES = ("scorer",)
    FUNCTION = "load_model"
    CATEGORY = "Scoring"

    def load_model(self, model_path: str = "") -> tuple[tuple[ScorerModel, Dict[str, Dict[str, int]]]]:
        # Resolve model path - prefer internal text model
        if not model_path:
            internal_path = os.path.join(os.path.dirname(__file__), "models", "text", "bin")
            model_path = internal_path

        if not os.path.isabs(model_path):
            base = os.path.dirname(__file__)
            potential = os.path.join(base, model_path)
            if os.path.exists(potential):
                model_path = potential

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path not found: {model_path}")

        maps_path = os.path.join(os.path.dirname(__file__), "models", "maps")
        maps = load_maps(maps_path)

        # Require node-local config
        node_root = os.path.dirname(__file__)
        candidate = os.path.join(node_root, "prepare_config.json")
        if not os.path.exists(candidate):
            raise FileNotFoundError(
                f"Required 'prepare_config.json' not found in node folder: {candidate}"
            )
        load_prepare_config(candidate)

        model = ScorerModel(model_path)
        return ((model, maps),)
