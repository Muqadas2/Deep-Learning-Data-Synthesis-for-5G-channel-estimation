import csv
import matplotlib.pyplot as plt
import os

csv_path = os.path.join("results", "summary_all_models.csv")

# Data containers
model_names = []
val_losses = []
params = []
inf_times = []

# Read CSV
with open(csv_path, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        model_names.append(row["Model"])
        val_loss = float(row["Final Val Loss"]) if row["Final Val Loss"] != "N/A" else None
        val_losses.append(val_loss)
        params.append(int(row["Total Params"]))
        inf_times.append(float(row["Avg Inference Time (s)"]))

# Plot
plt.figure(figsize=(10, 6))
bars = plt.bar(model_names, val_losses, color='skyblue', edgecolor='black')

# Add annotations
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.0002,
             f"{params[i]} params\n{inf_times[i]:.6f}s", ha='center', fontsize=8)

plt.ylabel("Final Validation MSE")
plt.title("Model Comparison (Validation MSE)")
plt.grid(True, axis='y', linestyle='--', alpha=0.6)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Save the plot
plot_path = os.path.join("results", "model_comparison.png")
plt.savefig(plot_path)
plt.show()

print(f"Model comparison plot saved to: {plot_path}")
