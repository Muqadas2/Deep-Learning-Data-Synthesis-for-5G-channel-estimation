import csv
import os
import matplotlib.pyplot as plt

# ====== File path ======
csv_path = os.path.join("results", "summary_all_models.csv")
plot_save_dir = "results"
os.makedirs(plot_save_dir, exist_ok=True)

# ====== Containers ======
model_names = []
val_losses = []
params = []
inf_times = []

# ====== Read CSV ======
with open(csv_path, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        model_names.append(row["Model"])
        val_losses.append(float(row["Final Val Loss"]))
        params.append(int(row["Total Params"]))
        inf_times.append(float(row["Avg Inference Time (s)"]))

# ====== Plot 1: Bar Chart - Val Loss vs Models ======
plt.figure(figsize=(10, 6))
bars = plt.bar(model_names, val_losses, color='skyblue', edgecolor='black')
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.002,
             f"{params[i]} params\n{inf_times[i]:.5f}s", ha='center', fontsize=8)
plt.ylabel("Final Validation MSE")
plt.title("Validation Loss vs Models")
plt.xticks(rotation=45, ha='right')
plt.grid(True, axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(plot_save_dir, "val_loss_vs_models.png"))
plt.show()

# ====== Plot 2: Bar Chart - Inference Time vs Models ======
plt.figure(figsize=(10, 6))
plt.bar(model_names, inf_times, color='orange', edgecolor='black')
plt.ylabel("Avg Inference Time (s)")
plt.title("Inference Time vs Models")
plt.xticks(rotation=45, ha='right')
plt.grid(True, axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(plot_save_dir, "inference_time_vs_models.png"))
plt.show()

# ====== Plot 3: Line Plot - Params vs Val Loss ======
plt.figure(figsize=(8, 5))
sorted_data = sorted(zip(params, val_losses, model_names))
sorted_params, sorted_losses, sorted_names = zip(*sorted_data)

plt.plot(sorted_params, sorted_losses, marker='o', linestyle='-', color='green')
for x, y, name in zip(sorted_params, sorted_losses, sorted_names):
    plt.annotate(name, (x, y), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8)

plt.xlabel("Total Parameters")
plt.ylabel("Final Validation MSE")
plt.title("Params vs Validation Loss")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(plot_save_dir, "params_vs_val_loss.png"))
plt.show()

# ====== Plot 4: Line Plot - Train/Val Loss Curves (Top 3 Models) ======
# Add your best 3 models' training logs here
top3_models = {
    "CNN_Original": "results/CNN_Original/training_log.csv",
    "CNN_OptimC_2": "results/CNN_OptimC_2/training_log.csv",
    "CNN_Merged": "results/CNN_Merged/training_log.csv"
}

plt.figure(figsize=(10, 6))
for model_name, log_path in top3_models.items():
    if not os.path.exists(log_path):
        print(f"Warning: {log_path} not found, skipping.")
        continue
    epochs = []
    train_loss = []
    val_loss = []
    with open(log_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["loss"]))
            val_loss.append(float(row["val_loss"]))
    plt.plot(epochs, train_loss, label=f"{model_name} - Train", linestyle='--')
    plt.plot(epochs, val_loss, label=f"{model_name} - Val")

plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.title("Training & Validation Loss Curves (Top 3 Models)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(plot_save_dir, "top3_loss_curves.png"))
plt.show()
