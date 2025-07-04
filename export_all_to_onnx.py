import os
import tensorflow as tf
import tf2onnx

# Directories
results_dir = "results"
onnx_dir = "onnx_exports"
os.makedirs(onnx_dir, exist_ok=True)

# Optional: per-model custom input shapes (batch size = 1 always)
model_input_shapes = {
    "Hybrid_CNN_Transformer_TF": (1, 612, 14, 1),  # Custom input shape
    # Add more model-specific shapes here if needed
}

# Default shape if not specified (TensorFlow format)
default_input_shape = (1, 612, 14, 1)


for model_folder in os.listdir(results_dir):
    model_name = model_folder
    keras_model_path = os.path.join(results_dir, model_folder, f"{model_name}_model.keras")
    onnx_output_path = os.path.join(onnx_dir, f"{model_name}.onnx")

    if not os.path.exists(keras_model_path):
        print(f"⚠️ Skipping {model_name}: .keras model file not found.")
        continue

    print(f"\n🔄 Converting: {model_name}")

    try:
        # Load model
        model = tf.keras.models.load_model(keras_model_path)

        # Set input shape with dynamic batch size
        input_shape = model_input_shapes.get(model_name, default_input_shape)
        dynamic_input_shape = [None] + list(input_shape[1:])  # Replace batch size with None

        inputs = tf.keras.Input(shape=dynamic_input_shape[1:], name="input")


        # If it's Sequential, wrap it
        if isinstance(model, tf.keras.Sequential):
            outputs = model(inputs)
            model = tf.keras.Model(inputs=inputs, outputs=outputs, name=model.name)

        # Convert to ONNX
        spec = (tf.TensorSpec(shape=dynamic_input_shape, dtype=tf.float32, name="input"),)

        model_proto, _ = tf2onnx.convert.from_keras(
            model,
            input_signature=spec,
            opset=13,
            output_path=onnx_output_path
        )

        print(f"✅ Saved: {onnx_output_path}")

    except Exception as e:
        print(f"❌ Failed to convert {model_name}: {e}")
