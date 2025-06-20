# models/model_optimC.py
import torch
import torch.nn as nn

class ChannelEstimationCNN(nn.Module):
    def __init__(self):
        super(ChannelEstimationCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(2),
            nn.ReLU(),

            nn.Conv2d(2, 2, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),

            nn.Conv2d(2, 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),

            nn.Conv2d(4, 1, kernel_size=1, padding=0)
        )

    def forward(self, x):
        return self.net(x)
