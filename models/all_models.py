import torch
import torch.nn as nn


def conv_bn_relu(in_channels, out_channels, kernel_size=3, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )
def depthwise_block(in_ch, out_ch, kernel_size=3, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=in_ch),
        nn.ReLU(),
        nn.Conv2d(in_ch, out_ch, kernel_size=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU()
    )

def conv_bn_silu(in_c, out_c, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size, stride, padding, dilation),
        nn.BatchNorm2d(out_c),
        nn.SiLU()
    )


class CNN_Original(nn.Module):
    """Basic CNN from original layers"""
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv2d(2, 2, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv2d(2, 2, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(2, 2, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(2, 1, kernel_size=5, padding=2)
        )

    def forward(self, x):
        return self.model(x)


class CNN_Optim(nn.Module):
    """Optimized CNN with 4-channel layers and BatchNorm"""
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            conv_bn_relu(1, 4),
            nn.Conv2d(4, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            conv_bn_relu(4, 4),
            nn.Conv2d(4, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 1, kernel_size=1)
        )

    def forward(self, x):
        return self.model(x)


class CNN_Merged(nn.Module):
    """Merged model with 5x5 filters and final projection"""
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            conv_bn_relu(1, 8, kernel_size=5, padding=2),
            conv_bn_relu(8, 8, kernel_size=5, padding=2),
            nn.Conv2d(8, 1, kernel_size=1)
        )

    def forward(self, x):
        return self.model(x)


class CNN_Depthwise(nn.Module):
    """Depthwise Separable CNN"""
    def __init__(self):
        super().__init__()

        def depthwise_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch),
                nn.ReLU(),
                nn.Conv2d(in_ch, out_ch, kernel_size=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU()
            )

        self.model = nn.Sequential(
            depthwise_block(1, 4),
            depthwise_block(4, 4),
            depthwise_block(4, 4),
            depthwise_block(4, 4),
            nn.Conv2d(4, 1, kernel_size=1)
        )

    def forward(self, x):
        return self.model(x)


class CNN_OptimDilation(nn.Module):
    """CNN with BatchNorm and no dilation"""
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            conv_bn_relu(1, 4),
            nn.Conv2d(4, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            conv_bn_relu(4, 4),
            nn.Conv2d(4, 1, kernel_size=1)
        )

    def forward(self, x):
        return self.model(x)


class CNN_OptimA(nn.Module):
    """Dilation only in first layer"""
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            conv_bn_relu(1, 4, dilation=2, padding=2),
            nn.Conv2d(4, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            conv_bn_relu(4, 4),
            nn.Conv2d(4, 1, kernel_size=1)
        )

    def forward(self, x):
        return self.model(x)


class CNN_OptimB(nn.Module):
    """Dilation + channel change + BN"""
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            conv_bn_relu(1, 4, dilation=2, padding=2),
            nn.Conv2d(4, 2, kernel_size=3, padding=1),
            nn.ReLU(),
            conv_bn_relu(2, 4),
            nn.Conv2d(4, 1, kernel_size=1)
        )

    def forward(self, x):
        return self.model(x)


class CNN_OptimC(nn.Module):
    """Double Dilation + Channel Expand"""
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            conv_bn_relu(1, 2, dilation=2, padding=2),
            nn.Conv2d(2, 2, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            conv_bn_relu(2, 4),
            nn.Conv2d(4, 1, kernel_size=1)
        )

    def forward(self, x):
        return self.model(x)
    
class CNN_OptimC_2(nn.Module):
    """Double Dilation + Channel Expand with InstanceNorm and SiLU"""
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            conv_bn_silu(1, 2, dilation=2, padding=2),              # (0)
            nn.Conv2d(2, 2, kernel_size=3, padding=2, dilation=2),  # (1)
            nn.SiLU(),                                              # (2)
            conv_bn_silu(2, 4),                                     # (3)
            nn.Conv2d(4, 1, kernel_size=1)                          # (4)
        )

    def forward(self, x):
        return self.model(x)

class CNN_OptimC_Depthwise(nn.Module):
    """
    OptimC with Depthwise Separable Convolutions + Dilation
    Replaces all standard convs with depthwise + pointwise
    """
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            depthwise_block(1, 2, dilation=2, padding=2),   # Input: 1 → 2
            depthwise_block(2, 2, dilation=2, padding=2),   # 2 → 2
            depthwise_block(2, 4),                          # 2 → 4
            nn.Conv2d(4, 1, kernel_size=1)                  # Final projection: 4 → 1
        )

    def forward(self, x):
        return self.model(x)

class CNN_OptimC_Depthwise_ResMix(nn.Module):
    """
    Updated OptimC-Depthwise: no dilation, residual + depthwise + standard conv mix
    """
    def __init__(self):
        super().__init__()
        
        self.block1 = depthwise_block(1, 2)               # 1 → 2
        self.block2 = conv_bn_relu(2, 2)                  # standard conv
        self.block3 = depthwise_block(2, 4)               # 2 → 4

        self.project_res = nn.Conv2d(2, 2, kernel_size=1) # for residual match

        self.final_conv = nn.Conv2d(4, 1, kernel_size=1)  # 4 → 1

    def forward(self, x):
        out1 = self.block1(x)               # (1 → 2)
        out2 = self.block2(out1)            # standard conv
        res = self.project_res(out1)        # match shape
        out2 = out2 + res                   # add residual connection
        out3 = self.block3(out2)            # (2 → 4)
        out = self.final_conv(out3)         # (4 → 1)
        return out
