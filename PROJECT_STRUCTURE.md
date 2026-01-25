# Project Structure
All output files are stored in `{module}/output/` directory.

## Workflow
1. Ranking (collect scores via UI)
2. Full data preparation (images + metadata → vectors + scores)
3. Text-only data preparation
4. Full data training (LightGBM → model export)
5. Text data training
6. Deploy node to ComfyUI

---

## Root-level files
- `config/config.json` — Main config (paths, file locations, node paths). Central source for `shared.config`.
- `config/prepare_config.json` — Prepare-time settings (vector schema, vision/text encoders, normalization).
- `config/training_config.json` — Training hyperparameter presets and HPO presets.
- `pyproject.toml` — Project dependencies and build metadata.
- `README.md` — Repo overview and usage instructions.
- `deploy.py` — Utility to copy the trained node and required assets to a ComfyUI node install (has safety checks).
- `.github/copilot-instructions.md` — Maintainer development instructions.
- `LICENSE`, `.gitignore`, `todo.md`, `done.md` — Repository housekeeping files.

---

## Shared (core utilities)
- `shared/config.py` — Config manager (`Config`) that enforces strict key presence and exposes `get`, `__getitem__`, `__setitem__`, `clear()`.
- `shared/io.py` — I/O helpers: `load_json`, `load_jsonl`, `atomic_write_json`, and vector serialization helpers used across prepare/training.
- `shared/utils.py` — Small helpers: `parse_custom_text`, `first_present`, `one_hot`, `binary_presence`, `weighted_presence`, and helpers used in tests.

---

## Ranking ( UI and HTTP service )
- `ranking/index.html` — Browser UI used to collect pairwise or single-image scores.
- `ranking/style.css` — Styles for the UI.
- `ranking/utils.py` — Helpers: `find_images(root)`, `get_json_path(img_path)`, `load_metadata(img_path)`, `get_unscored_images(root)`.
- `ranking/scores.py` — Server-side helpers: `_normalize_items`, `_write_score`, `submit_scores_handler` (atomic scoring updates).
- `ranking/score_server.py` — Flask-based server; endpoints: index, serve_image, random_unscored, submit_scores. Supports `--test-run` to validate configuration without starting server.

---

## Full data: prepare (image + metadata → vectors)
Top-level script
- `full_data/prepare/prepare_data.py` — Entry script and orchestration. Key functions:
    - `run_prepare(rebuild=False, limit=0)` — Collect + validate files → encode text/image → assemble vectors → append/save outputs.
    - `collect_valid_files(files, processed, error_log)` — Validate metadata and images; builds a list of items to encode.
    - `encode_new_images(collected_data)` — Uses configured vision and text encoders (lazy-loaded) to encode prompts and images.
    - `process_and_append_data(...)` — Build feature vectors and persist them in `vectors.jsonl`/`scores.jsonl` and `index.json`.

Prepare config & helpers
- `full_data/prepare/config/manager.py` — Loads the prepare config and vector schema with convenience functions.
- `full_data/prepare/config/maps.py` — Loads categorical maps used by `feature_assembler` (sampler, scheduler, model, lora maps).
- `full_data/prepare/config/schema.py` — Vector schema helpers: `get_vector_order()`, `get_slot_size()` and validation utilities.

Data ingest & validation
- `full_data/prepare/data/manager.py` — File collection (`collect_files`) and I/O glue to the encoding pipeline.
- `full_data/prepare/data/metadata.py` — `load_metadata_entry(path)` extracts the latest metadata entry and performs robust error handling.
- `full_data/prepare/data/processing.py` — Higher-level processing step that orchestrates encoding results and calls `process_and_append_data`.
- `full_data/prepare/data/text_processing.py` — Tokenization/text cleaning helpers used before embedding prompts.

Feature builders
- `full_data/prepare/features/assemble.py` — `assemble_feature_vector(...)` and helpers to produce a dense vector layout (concatenate scalar, embeddings, and categorical one-hots).
- `full_data/prepare/features/core.py` — Main feature extraction helpers used by `assemble_feature_vector`.
- `full_data/prepare/features/embeddings.py` — Wrappers around sentence-transformers and other text encoders (lazy-loaded to keep tests fast).
- `full_data/prepare/features/meta.py` — Extract metadata-derived features (size, aspect, model ids, sampler/scheduler indices).
- `full_data/prepare/features/terms.py` — Positive/negative term extraction utilities used by term-based features.
- `full_data/prepare/features/utils.py` — General helpers used across feature assembly.

Notes: Prepare code is designed to be test-friendly: heavy ML libs are lazy imported and places where real encoders are needed are easily stubbed in tests.

---

## Full data: training
- `full_data/training/run.py` — Orchestration for training and HPO. Important exported helpers:
    - `prepare_optimization_setup(base_cfg)` — Build search space and temp model paths.
    - `generate_combos(param_grid, max_combos)` — Randomized param combo generator for HPO.
    - `train_model_lightgbm_local(...)` — Core LightGBM training, metric computation, optional ONNX export.
    - `train_model(...)` — Top-level training orchestration: load data splits, call LightGBM, save model artifacts.
    - `optimize_hyperparameters(...)` — Runs HPO and updates `config['training']['top']` when improvements are found.
    - `compare_model_vs_data(...)` — Load saved model and produce analysis/plots.
- `full_data/training/data_utils.py` — Training-time data loading, splitting, and small utilities to prepare X,y for LightGBM.
- `full_data/training/model_io.py` — Model save/load helpers: `save_model_joblib`, `export_model_onnx`.
- `full_data/training/config_utils.py` — Grid and parameter composition utilities used when running HPO.
- `full_data/training/helpers.py` — Misc helpers used by notebooks and scripts (path utilities and plotting helpers).
- Notebooks: `training.ipynb`, `hyperparameter_optimize_loop_v2.ipynb`, `feature_importance_analysis.ipynb`, `polynomial_feature_experiment.ipynb`, `transformation_experiment.ipynb` — Interactive experiments and analysis.

Note: Training scripts attempt to import plotting libs (`matplotlib`) and may skip plotting in headless / CI environments; heavy ML libs are used in training but the repo uses lazy-loading and test shims to keep unit tests fast.

---

## Text data (text-only scoring pipeline)
- `text_data/prepare/prepare_text_data.py` — Entry script to prepare text-only features from stored prompts and text metadata.
- `text_data/prepare/text_processing.py` — Text normalization and tokenization helpers used by text-only pipeline.
- `text_data/prepare/__init__.py` — Package marker and small helpers.

---

## ComfyUI custom node (packaged node)
Directory: `comfyui_custom_nodes/ComfyUI-Image-Scorer`
- `__init__.py` — Exposes node package metadata.
- `feature_assembler.py` — `assemble_feature_vector(meta, pos_vec, neg_vec, categorical_indices)`: Node-compatible feature assembler; `load_maps` & `load_prepare_config` helpers.
- `nodes.py` — Node definitions (AestheticScoreNode, TextScoreNode) and loader helpers. Heavy ML imports are performed lazily to avoid import-time failures in ComfyUI or unit tests.
- `scorer.py` — `ScorerModel` wrapper to load ONNX/joblib scoring models and provide `predict`.
- `prepare_config.json` — Node-local prepare config used to ensure consistent vector layout inside the node.
- `README.md`, `requirements.txt` — Node documentation and node-specific dependencies.

---

## Typings used for static checks (non-executable)
- `typings/matplotlib/pyplot.pyi` — Type stubs for matplotlib.pyplot used in notebooks and helpers.
- `typings/sklearn/model_selection.pyi` — Type stubs for sklearn APIs.
- `typings/torch/*` — Minimal stubs used for type-checking in helper code.

---

## Notes on maintenance and testing ✅
- Unit tests are located alongside module tests and in `full_data/*/test` and `comfyui_custom_nodes/*/test`.
- Heavy dependencies (onnxruntime, sentence-transformers, transformers) are optional during unit tests — code uses lazy imports and small test shims.
- When making import changes, prefer editing the module that uses the dependency (do not add wrappers that only re-export imports).

---

If you'd like, I can now expand each module section further into per-function docstrings (method signatures and short descriptions) for every helper - or I can stop here and run the main files you asked for and fix any errors I find. Please confirm which next step you prefer. 🔧
