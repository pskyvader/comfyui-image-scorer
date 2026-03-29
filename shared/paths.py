import os
from .config import config
from pathlib import Path

root: Path = Path(__file__).parent.parent
config_dir: str = os.path.join(root, "config")
output_dir: str = os.path.join(root, "output")
maps_dir: str = os.path.join(output_dir, "maps")
cache_file: str = os.path.join(output_dir, "cache.db")

image_root: str = config["image_root"]

vectors_size_file: str = os.path.join(output_dir, "image_vector_size.json")
hyperparameters_statistics: str = os.path.join(output_dir, "hyperparameters_statistics.json")

vectors_dir: str = os.path.join(output_dir, "vectors")
vectors_file: str = os.path.join(vectors_dir, config["vectors_file"])
scores_file: str = os.path.join(vectors_dir, config["scores_file"])
index_file: str = os.path.join(vectors_dir, config["index_file"])
text_data_file: str = os.path.join(vectors_dir, config["text_data_file"])


models_dir: str = os.path.join(output_dir, "models")
training_model: str = os.path.join(models_dir, config["training_model"])
processed_data: str = os.path.join(models_dir, config["processed_data"])
filtered_data: str = os.path.join(models_dir, config["filtered_data"])
interaction_data: str = os.path.join(models_dir, config["interaction_data"])
