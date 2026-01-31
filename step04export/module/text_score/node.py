class TextScoreNode:
    """Scores text+params only using same vector assembly logic (no image features)."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "scorer": ("TEXT_SCORER",),
                "positive": ("STRING", {"multiline": True}),
                "negative": ("STRING", {"multiline": True}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 1000}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0}),
                "sampler": ("STRING", {"default": "euler"}),
                "scheduler": ("STRING", {"default": "normal"}),
                "model_name": ("STRING", {"default": "unknown"}),
                "lora_name": ("STRING", {"default": "unknown"}),
                "lora_strength": ("FLOAT", {"default": 0.0}),
                "width": ("INT", {"default": 512}),
                "height": ("INT", {"default": 512}),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("score",)
    FUNCTION = "calculate_score"
    CATEGORY = "Scoring"

    def __init__(self):
        # Lazy-load sentence transformer to avoid heavy imports at test time
        import types

        self.mpnet = types.SimpleNamespace()

    def _ensure_models_loaded(self):
        import types

        if isinstance(self.mpnet, types.SimpleNamespace):
            from sentence_transformers import SentenceTransformer

            self.mpnet = SentenceTransformer(MPNET_ID)

    def calculate_score(
        self,
        scorer: TextScorerLoader,
        positive: str,
        negative: str,
        steps: int,
        cfg: float,
        sampler: str,
        scheduler: str,
        model_name: str,
        lora_name: str,
        lora_strength: float,
        width: int,
        height: int,
    ) -> tuple[float]:
        if not (isinstance(scorer, tuple) and len(scorer) == 2):
            raise TypeError("'scorer' must be a tuple of (model, maps).")

        scorer_model, maps = scorer

        if not positive.strip():
            raise ValueError("The 'positive' prompt must be a non-empty string.")
        if not negative.strip():
            raise ValueError("The 'negative' prompt must be a non-empty string.")

        # Ensure mpnet is loaded lazily
        self._ensure_models_loaded()
        pos_vec = np.asarray(self.mpnet.encode(positive))
        neg_vec = np.asarray(self.mpnet.encode(negative))

        meta = {
            "steps": steps,
            "cfg": cfg,
            "width": width,
            "height": height,
            "lora_weight": lora_strength,
        }

        cat_indices = {
            "sampler": map_categorical_value(maps, "sampler", sampler),
            "scheduler": map_categorical_value(maps, "scheduler", scheduler),
            "model": map_categorical_value(maps, "model", model_name),
            "lora": map_categorical_value(maps, "lora", lora_name),
        }

        full_vec = assemble_feature_vector(meta, pos_vec, neg_vec, cat_indices)

        # Apply same filtering and interaction pipeline as image node expects
        model_path = getattr(scorer_model, "model_path", None)
        if model_path is None:
            model_path = getattr(scorer_model, "onnx_path", None)
        if model_path is None:
            model_bin_dir = os.path.join(os.path.dirname(__file__), "models", "bin")
        elif os.path.isfile(model_path):
            model_bin_dir = os.path.dirname(model_path)
        else:
            model_bin_dir = model_path

        filtered = apply_feature_filter([full_vec], model_bin_dir)
        final = apply_interaction_features(filtered, model_bin_dir)

        score = float(scorer_model.predict(final)[0])

        return (score,)
