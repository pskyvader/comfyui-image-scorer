import os
from .config import config
from pathlib import Path

# root = Path(config["root"])
root: Path = Path(__file__).parent.parent
config_dir: str = os.path.join(root, "config")
output_dir: str = os.path.join(root, "output")
maps_dir: str = os.path.join(output_dir, "maps")

image_root: str = config["image_root"]

vectors_size_file: str = os.path.join(config_dir, "image_vector_size.json")

vectors_dir: str = os.path.join(output_dir, "vectors")
vectors_file: str = os.path.join(vectors_dir, config["vectors_file"])
scores_file: str = os.path.join(vectors_dir, config["scores_file"])
index_file: str = os.path.join(vectors_dir, config["index_file"])


models_dir: str = os.path.join(output_dir, "models")
training_model: str = os.path.join(models_dir, config["training_model"])
processed_data: str = os.path.join(models_dir, config["processed_data"])
filtered_data: str = os.path.join(models_dir, config["filtered_data"])
interaction_data: str = os.path.join(models_dir, config["interaction_data"])


# print(f"root: {root}")
# print(f"output_dir: {output_dir}")
# print(f"maps_dir: {maps_dir}")


# comfy_node_path = config["comfy_node_path"]
# prepare_config = config["prepare_config"]
# training_config = config["training_config"]
# vector_config = config["vector_config"]


# print(f"image_root: {image_root}")
# print(f"comfy_node_path: {comfy_node_path}")
# print(f"prepare_config: {prepare_config}")
# print(f"config_path: {config_path}")
# print(f"training_config: {training_config}")
# print(f"vector_config: {vector_config}")


# prepare_dir = os.path.join(root, config["prepare_dir"])
# prepare_output_dir = os.path.join(prepare_dir, "output")


# error_log_file = os.path.join(
#     prepare_output_dir,
#     config["error_log_file"],
# )
# text_data_file = os.path.join(prepare_output_dir, config["text_data_file"])
# text_index_file = os.path.join(prepare_output_dir, config["text_index_file"])
# text_error_log_file = os.path.join(prepare_output_dir, config["text_error_log_file"])


# training_dir = os.path.join(root, config["training_dir"])
# training_output_dir = os.path.join(training_dir, "output")


# deployment_dir = os.path.join(root, config["deployment_dir"])
# deployment_module_dir = os.path.join(deployment_dir, "module")
