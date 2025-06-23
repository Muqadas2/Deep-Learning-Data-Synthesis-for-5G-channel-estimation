# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import csv
from datetime import datetime


def train(model, train_loader, val_loader, epochs=10, lr=3e-4, patience=5, val_freq=1,
          avg_time_per_sample=None, total_params=None, device='cuda'):

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    model_name = model.__class__.__name__
    output_dir = os.path.join("results", model_name)
    os.makedirs(output_dir, exist_ok=True)


    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    train_losses = []
    val_losses = []

    print("\nStarting training...\n")

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        if val_loader is not None and (epoch + 1) % val_freq == 0:
            model.eval()
            running_val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    pred = model(xb)
                    val_loss = criterion(pred, yb)
                    running_val_loss += val_loss.item()

            avg_val_loss = running_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                    break
        else:
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}")


    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print("\nTraining complete.\n")
    model_save_path = os.path.join(output_dir, f"{model_name}_model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved as: {model_save_path}")
    # ---- LOG FILE ----
    log_path = os.path.join(output_dir, "log.txt")
    with open(log_path, "a") as log_file:
        log_file.write(f"\n--- Training Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        log_file.write(f"Model: {model_name}\n")
        log_file.write(f"Epochs: {epochs}\n")
        log_file.write(f"Learning Rate: {lr}\n")
        log_file.write(f"Patience: {patience}\n")
        log_file.write(f"Validation Frequency: {val_freq}\n")
        log_file.write(f"Train Loss: {train_losses[-1]:.6f}\n")
        if val_losses:
            log_file.write(f"Val Loss: {val_losses[-1]:.6f}\n")
        log_file.write(f"Early Stopping Triggered: {'Yes' if patience_counter >= patience else 'No'}\n")
        log_file.write(f"Total Trainable Parameters: {total_params}\n")
        log_file.write(f"Avg Inference Time per Sample: {avg_time_per_sample:.6f} sec\n")
        log_file.write(f"Model Saved At: {model_save_path}\n")
        log_file.write("-" * 50 + "\n")

    print(f"Log file updated at: {log_path}")
        # ---- END LOG FILE ----
    

    # Save loss plot
    if val_loader is not None:
        model_name = model.__class__.__name__
        plot_filename = f"{model_name}_training_progress.png"
        plot_path = os.path.join(output_dir, f"{model_name}_training_progress.png")
        plt.figure(figsize=(8, 5))
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{model_name} - Training & Validation Loss")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        print(f"Training plot saved as: {plot_path}")

    else:
        print("No validation set provided, skipping validation loss plot.") 


    # Save results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f"{model_name}_training_results_{timestamp}.csv")
    with open(csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Train Loss", "Val Loss"])
        for i in range(len(train_losses)):
            val = val_losses[i] if i < len(val_losses) else ""
            writer.writerow([i+1, train_losses[i], val])

        writer.writerow([])
        writer.writerow(["Total Trainable Parameters", total_params])
        writer.writerow(["Avg Inference Time per Sample (s)", avg_time_per_sample])
        writer.writerow(["Model Architecture"])
        writer.writerow([str(model)])

    print(f"Training results saved to: {csv_path}")

    print("Training complete.") 

    # ---- APPEND TO GLOBAL SUMMARY FILE ----
    summary_path = os.path.join("results", "summary_all_models.csv")
    summary_exists = os.path.exists(summary_path)

    with open(summary_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not summary_exists:
            writer.writerow([
                "Timestamp", "Model", "Epochs", "LR", "Final Train Loss", "Final Val Loss",
                "Early Stopped", "Total Params", "Avg Inference Time (s)", "Model Path"
            ])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            model_name,
            epochs,
            lr,
            round(train_losses[-1], 6),
            round(val_losses[-1], 6) if val_losses else "N/A",
            "Yes" if patience_counter >= patience else "No",
            total_params,
            round(avg_time_per_sample, 6),
            model_save_path
        ])
        print(f"Global summary updated at: {summary_path}") 