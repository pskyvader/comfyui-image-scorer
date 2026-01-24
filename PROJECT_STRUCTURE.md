# Project Structure
All output files are stored in `{module}/output/` directory.

## workflow
# step 1, ranking
# step 2, full data process
# step 3, text data process
# step 4, full data training
# step 5, text data training
# step 6, deploy


## Root Level Files
- `config/config.json`: Main configuration file (consolidated into `config/` folder).
- pyproject.toml: Python dependencies (canonical dependency list).
- .github/copilot-instructions.md: Development instructions.
- PROJECT_STRUCTURE.md: This file.
- `config/prepare_config.json`: Configuration for data preparation.
- `config/training_config.json`: Configuration for model training.
- done.md: Completed tasks.
- todo.md: Pending tasks.
- deploy.py: Script to deploy the node to ComfyUI.


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

### full data
    prepare
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
            
    training
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


### text_data
    prepare
    training


### comfyui_custom_nodes
    ComfyUI-Image-Scorer/: