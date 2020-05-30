import torch
from torch import nn


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class RetinaHead(nn.Module):

    def __init__(self, in_channels, num_classes, anchor_ratios=[0.5, 1, 2]):
        super().__init__()
        self.anchor_ratios = anchor_ratios
        self.num_classes = num_classes
        self.double_conv = DoubleConv(in_channels, 128)
        self.out_class_conv = nn.Conv2d(128, self.num_classes * len(anchor_ratios), 3, 1, 1)

    def forward_single(self, x):
        x = self.double_conv(x)
        x = self.out_class_conv(x)
        x = torch.sigmoid(x)
        return x.view(x.shape[0], -1, self.num_classes)

    def forward(self, features, img_shape):
        # return [self.forward_single(x) for x in features][0]
        return torch.cat([self.forward_single(x) for x in features], 1)
