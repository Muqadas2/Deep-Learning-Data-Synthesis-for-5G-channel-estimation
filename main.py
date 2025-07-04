# main_tf.py
import os
import time
import numpy as np
import tensorflow as tf
from dataset import load_channel_dataset
from train import train
from training_configs import get_config
from models.top_models import CNN_OptimC_3_Tiny_Relu,CNN_OptimC_2,CNN_OptimC_2_Improved, CNN_OptimC_3,CNN_Original, CNN_OptimC_2_WithDenoiser, CNN_OptimC_2_WithDenoiser_PostRefined
import tensorflow as tf
tf.keras.backend.clear_session()

# List of model classes
all_model_classes = [ 
    CNN_OptimC_2,
    CNN_OptimC_2_Improved,
    CNN_OptimC_3_Tiny_Relu,
    CNN_OptimC_3,  
    CNN_OptimC_2_WithDenoiser,
    CNN_OptimC_2_WithDenoiser_PostRefined,
    CNN_Original,
    ]

# Load data
train_dataset = load_channel_dataset("data/tf_trainData.npy", "data/tf_trainLabels.npy", batch_size=8, shuffle=True)
val_dataset   = load_channel_dataset("data/tf_valData.npy", "data/tf_valLabels.npy", batch_size=8, shuffle=False)

# Device info
physical_devices = tf.config.list_physical_devices('GPU')
print(f"Using device: {'GPU' if physical_devices else 'CPU'}")

for ModelClass in all_model_classes:
    # Correctly get the model name from the class
    model_name = ModelClass.__name__

    # Instantiate the model
    model = ModelClass()

    # Match config
    if model_name == "CNN_Original":
        config_name = "base"
    elif model_name == "CNN_Merged":
        config_name = "merged"
    elif "Depthwise" in model_name:
        config_name = "depthwise"
    else:
        config_name = "optim"

    config = get_config(config_name)

    # Estimate inference time
    dummy_input = np.random.randn(1, 612, 14, 1).astype(np.float32)
    start = time.time()
    _ = model(dummy_input)
    end = time.time()
    avg_time_per_sample = (end - start)

    total_params = model.count_params()

    print(f"\nTraining {model_name} with config: {config}")
    train(
    model=model,
    model_name=model_name,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    epochs=config["epochs"],
    lr=config["lr"],
    patience=config["patience"],
    val_freq=config["val_freq"],
    avg_time_per_sample=avg_time_per_sample,
    total_params=total_params
    )


    # Create necessary directories
    os.makedirs("checkpoints", exist_ok=True)
    output_dir = os.path.join("results", model_name)
    os.makedirs(output_dir, exist_ok=True)

    # Save in .keras format
    keras_path_ckpt = os.path.join("checkpoints", f"{model_name}.keras")
    keras_path_out = os.path.join(output_dir, f"{model_name}_model.keras")
    model.save(keras_path_ckpt)
    model.save(keras_path_out)

    # Save in SavedModel format
    savedmodel_dir = os.path.join("checkpoints", f"{model_name}_SavedModel")
    model.export(savedmodel_dir)

    print(f" Model saved successfully!")
    print(f"- Native Keras checkpoint: {keras_path_ckpt}")
    print(f"- Keras copy in results:   {keras_path_out}")
    print(f"- TensorFlow SavedModel:   {savedmodel_dir}")
    print(f"- Total parameters:        {total_params}")
    print(f"- Avg inference time:      {avg_time_per_sample:.6f} seconds\n")
