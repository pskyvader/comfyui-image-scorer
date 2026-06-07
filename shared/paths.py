import os
from .config import config
from pathlib import Path

root: Path = Path(__file__).parents[1]
# config_dir: str = os.path.join(root, "config")
output_dir: str = os.path.join(root, "output")
maps_dir: str = os.path.join(output_dir, "maps")
cache_file: str = os.path.join(output_dir, "cache.db")

image_root: str = config["image_root"]
image_root_processed: str = os.path.join(image_root, "scored")

vectors_size_file: str = os.path.join(output_dir, "image_vector_size.json")
hyperparameters_statistics: str = os.path.join(
    output_dir, "hyperparameters_statistics.json"
)

vectors_dir: str = os.path.join(output_dir, "vectors")
split_dir: str = os.path.join(vectors_dir, "split")
vectors_file: str = os.path.join(vectors_dir, "vectors.jsonl")
scores_file: str = os.path.join(vectors_dir, "scores.jsonl")
comparisons_file: str = os.path.join(vectors_dir, "comparisons.jsonl")
index_file: str = os.path.join(vectors_dir, "index.jsonl")
text_data_file: str = os.path.join(vectors_dir, "text_data.jsonl")


models_dir: str = os.path.join(output_dir, "models")

training_model: str = os.path.join(models_dir, "model.npz")
raw_data: str = os.path.join(models_dir, "raw_data.npz")
comparison_data: str = os.path.join(models_dir, "comparison_data.npz")
filtered_data: str = os.path.join(models_dir, "filtered_data.npz")
interaction_data: str = os.path.join(models_dir, "interaction_data.npz")
