import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os


def train(model, train_loader, val_loader, epochs=10, lr=3e-4, patience=5, val_freq=1, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    train_losses = []
    val_losses = []

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

        # Validation
        if val_loader is not None and ((epoch + 1) % val_freq == 0):
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

            print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

            # Early stopping
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
            print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.6f}")

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Plot training vs validation loss
    if val_loader is not None:
        plt.figure(figsize=(8, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training & Validation Loss')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plot_path = "training_progress.png"
        plt.savefig(plot_path)
        print(f"Training progress plot saved as: {plot_path}")
        plt.show()
