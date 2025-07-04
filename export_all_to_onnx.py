import os
import tensorflow as tf
import tf2onnx

# Directories
results_dir = "results"
onnx_dir = "onnx_exports"
os.makedirs(onnx_dir, exist_ok=True)

# Optional: per-model custom input shapes (batch size = 1 by default)
model_input_shapes = {
    "Hybrid_CNN_Transformer_TF": (1, 612, 14, 1),
    # Add more model-specific shapes if needed
}

# Default input shape (TensorFlow format)
default_input_shape = (1, 612, 14, 1)

for model_folder in os.listdir(results_dir):
    model_name = model_folder
    keras_model_path = os.path.join(results_dir, model_folder, f"{model_name}_model.keras")
    onnx_output_path = os.path.join(onnx_dir, f"{model_name}.onnx")

    if not os.path.exists(keras_model_path):
        print(f"⚠ Skipping {model_name}: .keras model file not found.")
        continue

    print(f"\n🔄 Converting: {model_name}")

    try:
        # Load the Keras model
        model = tf.keras.models.load_model(keras_model_path)

        # Set input shape with dynamic batch size (None for batch)
        static_input_shape = model_input_shapes.get(model_name, default_input_shape)
        dynamic_input_shape = [None] + list(static_input_shape[1:])  # (None, 612, 14, 1)

        # Create new Input layer with dynamic shape
        inputs = tf.keras.Input(shape=dynamic_input_shape[1:], name="input")

        # If model is Sequential, wrap it in a functional model
        if isinstance(model, tf.keras.Sequential):
            outputs = model(inputs)
            model = tf.keras.Model(inputs=inputs, outputs=outputs, name=model.name)
        else:
            # Functional model: make sure input layer shape matches
            try:
                _ = model(inputs)
                model = tf.keras.Model(inputs=inputs, outputs=model(inputs), name=model.name)
            except:
                pass  # If already compatible, continue

        # Define input spec for ONNX export
        input_spec = (tf.TensorSpec(shape=dynamic_input_shape, dtype=tf.float32, name="input"),)

        # Convert to ONNX with dynamic_axes for batch
        model_proto, _ = tf2onnx.convert.from_keras(
            model,
            input_signature=input_spec,
            opset=13,
            output_path=onnx_output_path
        )


        print(f"✅ Saved: {onnx_output_path}")

    except Exception as e:
        print(f"❌ Failed to convert {model_name}: {e}")