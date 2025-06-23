import torch
import torch.nn as nn

class LayersOptimB(nn.Module):
    def __init__(self):
        super(LayersOptimB, self).__init__()
        self.net = nn.Sequential(
            # Input: (1, 624, 14)
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=2, dilation=2),  # Dilation=2, Padding=2 gives 'same' padding
            nn.BatchNorm2d(4),
            nn.ReLU(),

            nn.Conv2d(in_channels=4, out_channels=2, kernel_size=3, padding=1),  # Regular 3x3 conv
            nn.ReLU(),

            nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),

            nn.Conv2d(in_channels=4, out_channels=1, kernel_size=1, padding=0)  # 1x1 conv
        )

    def forward(self, x):
        return self.net(x)
