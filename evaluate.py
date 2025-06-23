import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import csv
from datetime import datetime
from models.all_models import CNN_Original, CNN_Optim, CNN_Merged, CNN_Depthwise, \
                              CNN_OptimDilation, CNN_OptimA, CNN_OptimB, CNN_OptimC, CNN_OptimC_Depthwise, CNN_OptimC_Depthwise_ResMix, CNN_OptimC_2

# List of model classes
all_model_classes = [
    # CNN_Original,
    # CNN_Optim,
    # CNN_Merged,
    # CNN_Depthwise,
    # CNN_OptimDilation,
    # CNN_OptimA,
    # CNN_OptimB,
    # CNN_OptimC,
    # CNN_OptimC_Depthwise,
    # CNN_OptimC_Depthwise_ResMix,
    CNN_OptimC_2,
]

# Load test data
data_dir = "data"
val_data_path = os.path.join(data_dir, "my_testData.npy")
val_labels_path = os.path.join(data_dir, "my_testLabels.npy")

val_data = np.load(val_data_path)        # shape: (N, C, H, W)
val_labels = np.load(val_labels_path)

val_data = torch.tensor(val_data, dtype=torch.float32)
val_labels = torch.tensor(val_labels, dtype=torch.float32)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
val_data = val_data.to(device)
val_labels = val_labels.to(device)

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
            "Timestamp",
            "Model",
            "Average MSE",
            "Total Samples",
            "Batch Inference Time (s)",
            "Avg Time per Sample (s)",
            "Single Sample Inference Time (s)",
            "Histogram Path"
        ])

    # Loop through all models
    for ModelClass in all_model_classes:
        model = ModelClass().to(device)
        model_name = model.__class__.__name__
        model_path = os.path.join("checkpoints", f"{model_name}.pth")

        if not os.path.exists(model_path):
            print(f"Skipping {model_name}: checkpoint not found at {model_path}")
            continue

        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        # Inference
        with torch.no_grad():
            start_time = time.time()
            predictions = model(val_data)
            end_time = time.time()

        # Single sample inference
        single_sample = val_data[0:1]
        with torch.no_grad():
            single_start = time.time()
            _ = model(single_sample)
            single_end = time.time()
        single_sample_time = single_end - single_start

        # Compute MSE
        mse_per_sample = torch.mean((predictions - val_labels) ** 2, dim=(1, 2, 3))
        avg_mse = mse_per_sample.mean().item()

        # Histogram
        hist_path = os.path.join(results_dir, f"mse_histogram_{model_name}.png")
        plt.figure(figsize=(8, 5))
        plt.hist(mse_per_sample.cpu().numpy(), bins=10, color='skyblue', edgecolor='black')
        plt.title(f"MSE Histogram - {model_name}")
        plt.xlabel("MSE")
        plt.ylabel("Number of Samples")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(hist_path)
        plt.close()

        # Print summary
        print(f"\nEvaluation Summary for {model_name}")
        print(f"-------------------------")
        print(f"Average MSE: {avg_mse:.6f}")
        print(f"Total Samples: {len(val_data)}")
        print(f"Inference Time (batch): {end_time - start_time:.4f} s")
        print(f"Avg Time per Sample: {(end_time - start_time)/len(val_data):.6f} s")
        print(f"Inference Time (single sample): {single_sample_time:.6f} s")
        print(f"Histogram saved to: {hist_path}")

        # Write to CSV
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = [
            timestamp,
            model_name,
            avg_mse,
            len(val_data),
            round(end_time - start_time, 6),
            round((end_time - start_time) / len(val_data), 6),
            round(single_sample_time, 6),
            os.path.abspath(hist_path)
        ]
        writer.writerow(row)

print("\n✅ Evaluation complete. All results saved.")
