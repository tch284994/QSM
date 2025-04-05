import torch
import torch.nn as nn
import torch.nn.functional as F


class ACAAttention(nn.Module):
    def __init__(self, channel, kernel_size=3):
        super(ACAAttention, self).__init__()
        self.channel = channel
        self.kernel_size = kernel_size

        padding = (kernel_size - 1) // 2
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(in_channels=channel, out_channels=channel, kernel_size=kernel_size, padding=padding, groups=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # Global average pooling and squeeze the size
        y = self.avg_pool(x)              # [bsz, channels, 1, 1]
        y = y.view(batch_size, channels)  # [bsz, channels]

        # Information interaction between adjacent channels through one-dimensional convolution
        y = self.conv(y.unsqueeze(2))     # [bsz, channels, 1]
        y = y.squeeze(2)                  # [bsz, channels]
        y = self.sigmoid(y)

        # Channel weighting of the original feature map
        y = y.view(batch_size, channels, 1, 1)  # [bsz, channels, 1, 1]
        return x * y.expand_as(x)
