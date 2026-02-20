# ComfyUI Image Aesthetic Scorer

This is a custom node for ComfyUI that predicts the aesthetic score of an image based on a trained LightGBM/ONNX model.

## Installation

1. Copy this folder `ComfyUI-Image-Scorer` into your ComfyUI `custom_nodes` directory.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   (Note: `onnxruntime-gpu` is recommended if you have CUDA. If not, use `onnxruntime`).

## Setup

You need to place your trained model files into the `training/output` folder (or point the node to wherever they are).
Required files:
- `model.onnx`
- `model.npz` (contains target transformation parameters)
- `processed_data_cache.npz` (contains feature filtering masks)

## Usage

1. **Load Aesthetic Scorer**: Use this node to load the model. Point `model_path` to the folder containing the files above.
2. **Calculate Aesthetic Score**: Connect the Scorer and an Image.
   - **Image**: Connect your generated image.
   - **Positive/Negative**: Connect string primitives.
   - **Config Values**: Set `steps`, `cfg`, `sampler` etc. matching your generation settings for best accuracy.

## Models Download

This node automatically downloads:
- Vision Model: `google/siglip-base-patch16-224`
- Text Model: `all-mpnet-base-v2`
(Cached in standard HuggingFace cache).
