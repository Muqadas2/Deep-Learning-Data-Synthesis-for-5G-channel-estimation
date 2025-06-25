import torch
import onnx
import os

from models.all_models import (  # Make sure this file contains all your models
    CNN_Original,
    CNN_Optim,
    CNN_Merged,
    CNN_Depthwise,
    CNN_OptimDilation,
    CNN_OptimA,
    CNN_OptimB,
    CNN_OptimC,
    CNN_OptimC_2,
    CNN_OptimC_Depthwise,
    CNN_OptimC_Depthwise_ResMix
)

# Dummy input shape — change if needed
dummy_input = torch.randn(1, 1, 14, 624)  # (batch_size, channels, height, width)

# Model classes and names
models = {
    "cnn_original": CNN_Original,
    "cnn_optim": CNN_Optim,
    "cnn_merged": CNN_Merged,
    "cnn_depthwise": CNN_Depthwise,
    "cnn_optim_dilation": CNN_OptimDilation,
    "cnn_optim_a": CNN_OptimA,
    "cnn_optim_b": CNN_OptimB,
    "cnn_optim_c": CNN_OptimC,
    "cnn_optim_c_2": CNN_OptimC_2,
    "cnn_optim_c_depthwise": CNN_OptimC_Depthwise,
    "cnn_optim_c_depthwise_resmix": CNN_OptimC_Depthwise_ResMix
}

# Output directory
os.makedirs("onnx_exports", exist_ok=True)

# Export each model
for name, model_class in models.items():
    print(f"Exporting {name}.onnx ...")

    model = model_class()
    model.eval()

    onnx_path = f"onnx_exports/{name}.onnx"

    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            opset_version=11
        )

        # Validate ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print(f"Exported and validated: {onnx_path}")

    except Exception as e:
        print(f"Failed to export {name}: {e}")
