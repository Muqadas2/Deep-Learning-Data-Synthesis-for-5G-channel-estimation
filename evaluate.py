# eval_tf.py
import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import csv
from datetime import datetime
from models.all_models import CNN_Original, CNN_Optim, CNN_Merged, CNN_Depthwise, \
                                  CNN_OptimDilation, CNN_OptimA, CNN_OptimB, CNN_OptimC, CNN_OptimC_Depthwise, CNN_OptimC_Depthwise_ResMix, CNN_OptimC_2, Hybrid_CNN_Transformer_TF

# List of model classes
all_model_classes = [
    CNN_Original,
    CNN_Optim,
    CNN_Merged,
    CNN_Depthwise,
    CNN_OptimDilation,
    CNN_OptimA,
    CNN_OptimB,
    CNN_OptimC,
    CNN_OptimC_Depthwise,
    CNN_OptimC_Depthwise_ResMix,
    CNN_OptimC_2,
    Hybrid_CNN_Transformer_TF,
    # Lightweight_DNN_CIR
]

# Load test data
val_data = np.load("data/tf_testData.npy")
val_labels = np.load("data/tf_testLabels.npy")


# Results
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

# CSV for all evaluations
csv_path = os.path.join(results_dir, "eval_results.csv")
file_exists = os.path.isfile(csv_path)

with open(csv_path, mode="a", newline="") as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow([
            "Timestamp", "Model", 
            "Average MSE", "NMSE", "NMSE (dB)", "Total Samples",
            "Batch Inference Time (s)", "Avg Time per Sample (s)",
            "Single Sample Inference Time (s)", "Histogram Path"
        ])

    for ModelClass in all_model_classes:
        model = ModelClass()
        model_name = ModelClass.__name__
        model_path = os.path.join("checkpoints", f"{model_name}.keras")

        if not os.path.exists(model_path):
            print(f"Skipping {model_name}: checkpoint not found at {model_path}")
            continue

        model = tf.keras.models.load_model(model_path)

        # Flatten input if needed
        if model_name == "Lightweight_DNN_CIR":
            val_data = val_data.reshape(-1, 612 * 14)
            val_labels = val_labels.reshape(-1, 612 * 14)

        # Inference
        start = time.time()
        predictions = model.predict(val_data)
        end = time.time()

        # Compute MSE per sample
        if model_name == "Lightweight_DNN_CIR":
            predictions = predictions.reshape(-1, 612 * 14)
            mse_per_sample = np.mean((predictions - val_labels) ** 2, axis=1)
        else:
            mse_per_sample = np.mean((predictions - val_labels) ** 2, axis=(1, 2, 3))

        avg_mse = np.mean(mse_per_sample)

        # Compute NMSE
        numerator = np.sum((predictions - val_labels) ** 2)
        denominator = np.sum(val_labels ** 2)
        nmse = numerator / denominator
        nmse_db = 10 * np.log10(nmse)

        # Inference time per sample
        single_start = time.time()
        _ = model.predict(val_data[0:1])
        single_end = time.time()
        single_sample_time = single_end - single_start

        # Histogram
        hist_path = os.path.join(results_dir, f"mse_histogram_{model_name}.png")
        plt.figure(figsize=(8, 5))
        plt.hist(mse_per_sample, bins=10, color='skyblue', edgecolor='black')
        plt.title(f"MSE Histogram - {model_name}")
        plt.xlabel("MSE")
        plt.ylabel("Number of Samples")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(hist_path)
        plt.close()

        # Print results
        print(f"\nEvaluation Summary for {model_name}")
        print("-------------------------")
        print(f"Average MSE: {avg_mse:.6f}")
        print(f"NMSE: {nmse:.6f}")
        print(f"NMSE (dB): {nmse_db:.2f} dB")
        print(f"Total Samples: {len(val_data)}")
        print(f"Inference Time (batch): {end - start:.4f} s")
        print(f"Avg Time per Sample: {(end - start)/len(val_data):.6f} s")
        print(f"Inference Time (single sample): {single_sample_time:.6f} s")
        print(f"Histogram saved to: {hist_path}")

        # Save to CSV
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([
            timestamp, model_name, 
            round(avg_mse, 8), round(nmse, 8), round(nmse_db, 4),
            len(val_data),
            round(end - start, 6),
            round((end - start)/len(val_data), 6),
            round(single_sample_time, 6),
            os.path.abspath(hist_path)
        ])

print("\n✅ Evaluation complete. All results saved.")
