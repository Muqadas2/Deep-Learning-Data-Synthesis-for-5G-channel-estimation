import os
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf

# Paths
onnx_dir = "onnx_exports"
tf_dir = "tf_exports"
tflite_dir = "tflite_exports"

# Create output dirs if not exist
os.makedirs(tf_dir, exist_ok=True)
os.makedirs(tflite_dir, exist_ok=True)

# Get list of .onnx files
onnx_files = [f for f in os.listdir(onnx_dir) if f.endswith(".onnx")]

for onnx_file in onnx_files:
    model_name = onnx_file.replace(".onnx", "")
    onnx_path = os.path.join(onnx_dir, onnx_file)
    
    print(f"Converting {onnx_file} ...")

    # Load ONNX model
    onnx_model = onnx.load(onnx_path)

    # Convert to TensorFlow SavedModel
    tf_model = prepare(onnx_model)
    tf_model_path = os.path.join(tf_dir, model_name + "_tf")
    tf_model.export_graph(tf_model_path)

    print(f"Saved TensorFlow model: {tf_model_path}")

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
    tflite_model = converter.convert()

    # Save TFLite model
    tflite_path = os.path.join(tflite_dir, model_name + ".tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    print(f"Saved TFLite model: {tflite_path}\n")

print(" All models converted successfully.")
