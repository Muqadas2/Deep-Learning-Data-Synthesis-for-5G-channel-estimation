import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import csv
from datetime import datetime

# === Config ===
helena_model_dir = "helena_models"  # folder containing .keras models
val_data_path = "data/tf_testData.npy"
val_labels_path = "data/tf_testLabels.npy"
results_dir = "results_helena"
csv_path = os.path.join(results_dir, "eval_helena_results.csv")

os.makedirs(results_dir, exist_ok=True)

# === Load data ===
val_data = np.load(val_data_path)
val_labels = np.load(val_labels_path)

# === Start writing results ===
file_exists = os.path.exists(csv_path)

with open(csv_path, mode="a", newline="") as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow([
            "Timestamp", "Model",
            "Average MSE", "NMSE", "NMSE (dB)", "Total Samples",
            "Batch Inference Time (s)", "Avg Time per Sample (s)",
            "Single Sample Inference Time (s)", "Histogram Path"
        ])

    # Loop over all .keras models
    for model_file in os.listdir(helena_model_dir):
        if not model_file.endswith(".keras"):
            continue

        model_path = os.path.join(helena_model_dir, model_file)
        model_name = os.path.splitext(model_file)[0]
        print(f"\n🔍 Evaluating: {model_name}")

        # Load model
        model = tf.keras.models.load_model(model_path, compile=False)


        # Inference (batch)
        start = time.time()
        predictions = model.predict(val_data, verbose=0)
        end = time.time()

        # MSE per sample
        mse_per_sample = np.mean((predictions - val_labels) ** 2, axis=(1, 2, 3))
        avg_mse = np.mean(mse_per_sample)

        # NMSE
        numerator = np.sum((predictions - val_labels) ** 2)
        denominator = np.sum(val_labels ** 2)
        nmse = numerator / denominator
        nmse_db = 10 * np.log10(nmse)

        # Inference time (single sample)
        single_start = time.time()
        _ = model.predict(val_data[0:1], verbose=0)
        single_end = time.time()
        single_sample_time = single_end - single_start

        # Histogram
        hist_path = os.path.join(results_dir, f"mse_histogram_{model_name}.png")
        plt.figure(figsize=(8, 5))
        plt.hist(mse_per_sample, bins=10, color='salmon', edgecolor='black')
        plt.title(f"MSE Histogram - {model_name}")
        plt.xlabel("MSE")
        plt.ylabel("Number of Samples")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(hist_path)
        plt.close()

        # Console log
        print(f"✅ Avg MSE: {avg_mse:.6f}")
        print(f"📉 NMSE: {nmse:.6f} ({nmse_db:.2f} dB)")
        print(f"🧪 Inference (batch): {end - start:.4f} s, single: {single_sample_time:.6f} s")

        # Save row to CSV
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([
            timestamp, model_name,
            round(avg_mse, 8), round(nmse, 8), round(nmse_db, 4),
            len(val_data),
            round(end - start, 6),
            round((end - start) / len(val_data), 6),
            round(single_sample_time, 6),
            os.path.abspath(hist_path)
        ])

print("\n✅ HELENA model evaluation complete. All results saved.")
