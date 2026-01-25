from pathlib import Path
import sys
# Ensure repo root is on sys.path so top-level packages (shared, full_data, etc.) can be imported
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
# Also ensure full_data is on sys.path so we can import the training package
sys.path.insert(0, str(root / 'full_data'))
from shared.config import config
from training.helpers import resolve_path
from training.data_utils import load_training_data, filter_unused_features, add_interaction_features
from training.config_utils import generate_random_config
from training.run import prepare_optimization_setup, generate_combos, evaluate_hyperparameter_combo

# Setup
config["training"]["live_graph_path"] = "training/output/graph_hpo.png"
print('Live plot set to:', resolve_path(config["training"]["live_graph_path"]))

vectors_path = resolve_path(config["vectors_file"])
scores_path = resolve_path(config["scores_file"])
print('Loading data from:', vectors_path, scores_path)
X, y = load_training_data(vectors_path, scores_path)
print('Loaded:', X.shape)

# Filter
X, kept = filter_unused_features(X, y)
print('Filtered shape:', X.shape)

# Add interactions
X, _ = add_interaction_features(X, y, target_k=50)
print('After interactions:', X.shape)

# Prepare optimization setup
base_cfg = generate_random_config()
param_grid, temp_model_base = prepare_optimization_setup(base_cfg)
print('Param grid keys:', list(param_grid.keys()))
combos = generate_combos(param_grid, max_combos=2)
print('Generated combos:', combos)

# Evaluate one combo quickly (uses lightgbm)
if combos:
    score, t_time = evaluate_hyperparameter_combo(combos[0], temp_model_base, X, y)
    print('Eval combo score, time:', score, t_time)
else:
    print('No combos generated.')
