import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from models.model_base import CNNModel

# Paths
data_dir = "data"
val_data_path = os.path.join(data_dir, "my_testData.npy")
val_labels_path = os.path.join(data_dir, "my_testLabels.npy")
model_path = "channel_estimator_base.pth"

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
val_data = np.load(val_data_path)        # shape: (8, 1, 612, 14)
val_labels = np.load(val_labels_path)    # shape: (8, 1, 612, 14)

# Convert to torch tensors
val_data = torch.tensor(val_data, dtype=torch.float32).to(device)
val_labels = torch.tensor(val_labels, dtype=torch.float32).to(device)

# Load model
model = CNNModel().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Inference
with torch.no_grad():
    import time
    start_time = time.time()
    predictions = model(val_data)
    end_time = time.time()

# Compute MSE per sample
mse_per_sample = torch.mean((predictions - val_labels) ** 2, dim=(1, 2, 3))
avg_mse = mse_per_sample.mean().item()

# Histogram plot
plt.figure(figsize=(8, 5))
plt.hist(mse_per_sample.cpu().numpy(), bins=10, color='skyblue', edgecolor='black')
plt.title("MSE over test Samples")
plt.xlabel("MSE")
plt.ylabel("Number of Samples")
plt.grid(True)
plt.tight_layout()
plt.savefig("mse_histogram.png")
plt.show()

# Print summary
print(f"Evaluation Summary")
print(f"-------------------------")
print(f"Average MSE: {avg_mse:.6f}")
print(f"Total Samples: {len(val_data)}")
print(f"Inference Time (batch): {end_time - start_time:.4f} s")
print(f"Time per Sample: {(end_time - start_time)/len(val_data):.6f} s")
print(f"Histogram saved to: mse_histogram.png")
