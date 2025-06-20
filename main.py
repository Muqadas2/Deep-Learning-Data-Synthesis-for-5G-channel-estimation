# main.py
import torch
from models.model_base import CNNModel
from dataset import ChannelDataset
from train import train
from torch.utils.data import DataLoader

train_ds = ChannelDataset("data/my_trainData.npy", "data/my_trainLabels.npy")
val_ds = ChannelDataset("data/my_valData.npy", "data/my_valLabels.npy")

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=8)

model = CNNModel()
train(model, train_loader, val_loader, epochs=10)

# Save the trained model
torch.save(model.state_dict(), "channel_estimator_base.pth")
# Print a message indicating that training is complete
print("Training complete. Model saved as 'channel_estimator_base.pth'.")

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params}")
# Print the model architecture
print(model)

