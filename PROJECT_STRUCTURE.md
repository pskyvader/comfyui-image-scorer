# Project Structure
All output files are stored in `{module}/output/` directory.

## Root Level Files
- config.json: Main configuration file.
- requirements.txt: Python dependencies (canonical dependency list).
- .github/copilot-instructions.md: Development instructions.
- PROJECT_STRUCTURE.md: This file.
- prepare_config.json: Configuration for data preparation.
- training_config.json: Configuration for model training.
- done.md: Completed tasks.
- todo.md: Pending tasks.
- deploy.py: Script to deploy the node to ComfyUI.
- conftest.py: Pytest configuration.

## Modules

### shared
- config.py: Configuration loading and management.
- io.py: JSON/JSONL I/O utilities (atomic write, robust load, vector extraction).
- utils.py: Shared utilities.

### ranking
- score_server.py: Flask server for image ranking.
- scores.py: Score submission handling.
- utils.py: Image finding and metadata loading utilities.
- index.html: Ranking UI.
- style.css: Ranking UI styles.

### prepare
- prepare_data.py: Main script for image data preparation.
- config/
    - manager.py: Config access.
    - maps.py: Categorical mapping loading.
    - schema.py: Vector schema definitions.
- data/
    - manager.py: Image loading and CLIP encoding.
    - metadata.py: Metadata loading and error logging.
    - processing.py: Data processing pipeline.
    - text_processing.py: Text processing logic (for prepare module).
- features/
    - assemble.py: Feature vector assembly.
    - core.py: Core feature extraction.
    - embeddings.py: Text embedding utilities.
    - meta.py: Metadata parsing.
    - terms.py: Term extraction.
    - utils.py: Feature utilities.

### text_data
- prepare_text_data.py: Main script for text data export.
- text_processing.py: Text processing logic.
- __init__.py: Module initialization.

### training
- run.py: Main training logic (LightGBM, optimization, ONNX export).
- text/: Placeholder and tools for training text-only scoring models.
- config_utils.py: Configuration utilities and grid search definitions.
- data_utils.py: Training data loading and splitting.
- helpers.py: General helpers (path resolution).
- model_io.py: Model saving/loading (ONNX/Joblib).
- training.ipynb: Interactive training notebook.
- hyperparameter_optimize_loop_v2.ipynb: HPO notebook.
- feature_importance_analysis.ipynb: Feature analysis notebook.
- polynomial_feature_experiment.ipynb: Polynomial feature experiments.
- transformation_experiment.ipynb: Transformation experiments.

### comfyui_custom_nodes
- ComfyUI-Image-Scorer/:
    - __init__.py: Node mappings export.
    - nodes.py: Main node logic (ScorerLoader, AestheticScoreNode, TextScoreNode).
    - requirements.txt: Node dependencies.
    - README.md: Installation guide.
    - prepare_config.json: Node-local configuration required by the node at runtime.
    - lib/:
        - feature_assembler.py: Feature vector construction logic (independent of project config).
        - scorer.py: ONNX Runtime scoring wrapper with inverse transforms.
    - models/:
        - bin/: Contains bundled ONNX model and cache files.
        - text/bin/: Placeholder location for text-only model files.
        - maps/: Contains categorical mapping files.

## Tests
- prepare/test/:
    - test_data_manager.py: Tests for data manager.
    - test_data_metadata.py: Tests for metadata.
    - test_data_processing.py: Tests for data processing.
    - test_data_text_processing.py: Tests for text processing.
    - test_features.py: Tests for feature assembly.
    - test_features_core.py: Tests for core features.
    - test_features_embeddings.py: Tests for embeddings.
    - test_features_meta.py: Tests for metadata features.
    - test_features_terms.py: Tests for terms features.
    - test_features_utils.py: Tests for feature utils.
    - test_prepare_config.py: Tests for config manager.
- ranking/test/: 
    - test_scores.py: Tests for scores module.
    - test_utils.py: Tests for ranking utilities.
- shared/test/: 
    - test_io.py: Tests for IO module.
    - test_shared_config.py: Tests for shared config.
    - test_shared_utils_unique.py: Tests for shared utils.
- text_data/test/: 
    - test_text_processing.py: Tests for text processing.
- training/test/: 
    - test_config_utils.py: Tests for config utils.
    - test_data_utils.py: Tests for data utils.
    - test_helpers.py: Tests for helpers.
    - test_model_io.py: Tests for model IO.
    - test_run.py: Tests for run module.
    - test_run_around.py: Integration tests for run module.
- test_custom_nodes/:
    - test_nodes.py: Tests for generic node logic.
    - test_lib_feature_assembler.py: Tests for feature assembler lib.
    - test_lib_scorer.py: Tests for scorer lib.
