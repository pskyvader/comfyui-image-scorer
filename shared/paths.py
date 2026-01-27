import os
from shared.config import config
from pathlib import Path

print(f"config: {config}")
root = Path(config["root"])
print(f"root: {root}")
image_root = os.path.join(config["image_root"], "/")
comfy_node_path = os.path.join(config["comfy_node_path"], "/")
prepare_config = os.path.join(config["prepare_config"], "/")

prepare_dir = os.path.join(root, config["prepare_dir"], "/")
print(f"prepare_dir: {prepare_dir}")
prepare_output_dir = os.path.join(prepare_dir, "output")
maps_dir = os.path.join(prepare_dir, "full_data", "maps")


vectors_file = os.path.join(prepare_output_dir, config["vectors_file"])
scores_file = os.path.join(prepare_output_dir, config["scores_file"])
index_file = os.path.join(prepare_output_dir, config["index_file"])
error_log_file = os.path.join(
    prepare_output_dir,
    config["error_log_file"],
)
text_data_file = os.path.join(prepare_output_dir, config["text_data_file"])
text_index_file = os.path.join(prepare_output_dir, config["text_index_file"])
text_error_log_file = os.path.join(prepare_output_dir, config["text_error_log_file"])


training_dir = os.path.join(root, config["training_dir"], "/")
training_output_dir = os.path.join(training_dir, "output")

training_model = os.path.join(training_output_dir, config["training_model"])
processed_data = os.path.join(training_output_dir, config["processed_data"])
filtered_data = os.path.join(training_output_dir, config["filtered_data"])
interaction_data = os.path.join(training_output_dir, config["interaction_data"])


deployment_dir = os.path.join(root, config["deployment_dir"], "/")
deployment_module_dir = os.path.join(deployment_dir, "module")
