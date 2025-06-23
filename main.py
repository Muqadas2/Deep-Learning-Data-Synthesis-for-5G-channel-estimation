# main.py
import torch
import time
from dataset import ChannelDataset
from train import train
from torch.utils.data import DataLoader
from training_configs import get_config
import os
from models.all_models import CNN_Original, CNN_Optim, CNN_Merged, CNN_Depthwise, \
                              CNN_OptimDilation, CNN_OptimA, CNN_OptimB, CNN_OptimC, CNN_OptimC_Depthwise, CNN_OptimC_Depthwise_ResMix, CNN_OptimC_2
from models.model_optimC import ChannelEstimationCNN
from training_configs import get_config

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
    # ChannelEstimationCNN,
    CNN_OptimC_2,
]


# Load data
train_ds = ChannelDataset("data/my_trainData.npy", "data/my_trainLabels.npy")
val_ds = ChannelDataset("data/my_valData.npy", "data/my_valLabels.npy")

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=8)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

for ModelClass in all_model_classes:
    model = ModelClass()
    model_name = model.__class__.__name__

    # Map model name to config type
    if model_name == "CNN_Original":
        config_name = "base"
    elif model_name == "CNN_Merged":
        config_name = "merged"
    elif model_name == "CNN_Depthwise" or model_name.endswith("Depthwise"):
        config_name = "depthwise"
    else:
        config_name = "optim"

    config = get_config(config_name)

    # Estimate inference time per sample
    dummy_input = next(iter(train_loader))[0].to(device)
    with torch.no_grad():
        start = time.time()
        _ = model(dummy_input)
        end = time.time()
    avg_time_per_sample = (end - start) / dummy_input.size(0)

    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nTraining {model_name} with config: {config}")
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config["epochs"],
        lr=config["lr"],
        patience=config["patience"],
        val_freq=config["val_freq"],
        avg_time_per_sample=avg_time_per_sample,
        total_params=total_params,
        device=device
    )

    os.makedirs("checkpoints", exist_ok=True)
    save_path = os.path.join("checkpoints", f"{model_name}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to: {save_path}")
