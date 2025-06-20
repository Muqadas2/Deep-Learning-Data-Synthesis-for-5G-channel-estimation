import torch
import torch.nn as nn

# Define your CNNModel
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=9, padding=4),  # same padding
            nn.ReLU(),

            nn.Conv2d(2, 2, kernel_size=9, padding=4),
            nn.ReLU(),

            nn.Conv2d(2, 2, kernel_size=5, padding=2),
            nn.ReLU(),

            nn.Conv2d(2, 2, kernel_size=5, padding=2),
            nn.ReLU(),

            nn.Conv2d(2, 1, kernel_size=5, padding=2)   # output layer
        )
    def forward(self, x):
        return self.net(x)

# Instantiate the model and load weights
model = CNNModel()
model.load_state_dict(torch.load("/models/model_base.py"))  # Make sure this matches your file
model.eval()

# Dummy input with correct shape: (batch_size, channels, height, width)
dummy_input = torch.randn(1, 1, 624, 14)  # Assuming 624x14 image input for your case

# Export to ONNX
torch.onnx.export(model,
                  dummy_input,
                  "cnn_model.onnx",
                  input_names=["input"],
                  output_names=["output"],
                  opset_version=11)

print("✅ Exported to cnn_model.onnx")
