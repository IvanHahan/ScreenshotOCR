import torch
from torch import nn
import numpy as np


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
        self.class_branch = nn.Sequential(
            DoubleConv(in_channels, 128),
            nn.Conv2d(128, num_classes * len(anchor_ratios), 3, 1, 1),
            nn.Sigmoid()
        )
        self.boxes_branch = nn.Sequential(
            DoubleConv(in_channels, 128),
            nn.Conv2d(128, 4 * len(anchor_ratios), 3, 1, 1),
        )

    def forward_single(self, x, img_shape):
        classes = self.class_branch(x)
        classes = classes.view(x.shape[0], -1, self.num_classes)

        boxes = self.boxes_branch(x)
        boxes = boxes.permute(0, 2, 3, 1)
        boxes[..., 0::4] = torch.sigmoid(boxes[..., 0::4])
        boxes[..., 1::4] = torch.sigmoid(boxes[..., 1::4])

        cell_shape = img_shape // np.array(list(boxes.shape[1:3]))
        anchors = []
        for ratio in self.anchor_ratios:
            anchor = [max(cell_shape), max(cell_shape)]
            if ratio < 1:
                anchor[0] /= ratio
            else:
                anchor[1] *= ratio
            anchors.append(anchor)

        output_boxes = boxes.clone()
        output_boxes[..., 0::4] = output_boxes[..., 0::4] * cell_shape[1] + \
                                  torch.arange(0, boxes.shape[2]).repeat(boxes.shape[1], 1) \
                                      .view(1, boxes.shape[1], boxes.shape[2], 1) * cell_shape[1]
        output_boxes[..., 1::4] = output_boxes[..., 1::4] * cell_shape[0] + \
                                  torch.arange(0, boxes.shape[1]).repeat(boxes.shape[2], 1).t() \
                                      .view(1, boxes.shape[1], boxes.shape[2], 1) * cell_shape[0]
        for i, anchor in enumerate(anchors):
            output_boxes[..., (i * 4) + 2] = torch.exp(output_boxes[..., (i * 4) + 2]) * anchor[1]
            output_boxes[..., (i * 4) + 3] = torch.exp(output_boxes[..., (i * 4) + 3]) * anchor[0]

        output_boxes = output_boxes.contiguous().view(output_boxes.size(0), -1, 4)
        train_boxes = boxes.contiguous().view(output_boxes.size(0), -1, 4)

        return classes, train_boxes, output_boxes

    def forward(self, features, img_shape):
        # return [self.forward_single(x) for x in features][0]
        classes, train_boxes, output_boxes = list(zip(*[self.forward_single(x, img_shape) for x in features[:1]]))
        classes = torch.cat(classes, 1)
        train_boxes = torch.cat(train_boxes, 1)
        output_boxes = torch.cat(output_boxes, 1)
        return classes, train_boxes, output_boxes
