import torch
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=2, kernel_size=9, padding=4),  # same padding
            nn.ReLU(),

            nn.Conv2d(2, 2, kernel_size=9, padding=4),
            nn.ReLU(),

            nn.Conv2d(2, 2, kernel_size=5, padding=2),
            nn.ReLU(),

            nn.Conv2d(2, 2, kernel_size=5, padding=2),
            nn.ReLU(),

            nn.Conv2d(2, 1, kernel_size=5, padding=2)  # final output layer
        )
    
    def forward(self, x):
        return self.net(x)
