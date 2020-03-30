import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(
        a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(
        a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) *
                         (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih

    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    return IoU


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma

    def forward(self, predictions, target):
        # predictions batch x boxes x (cx, cy, w, h, conf)
        # target batch x boxes x (cx, cy, w, h, conf)
        # anchors batch x boxes x (w, h)

        pred_conf = predictions[..., -1]
        pred_boxes = predictions[..., :4]

        target_conf = target[..., -1]
        target_boxes = target[..., :4]

        bce = F.binary_cross_entropy(pred_conf, target_conf)
        pt = torch.exp(-bce)
        f_loss = self.alpha * (1 - pt) ** self.gamma * bce

        object_mask = target_conf == 1

        object_target_boxes = target_boxes[object_mask]
        object_pred_boxes = pred_boxes[object_mask]

        localization_loss = F.mse_loss(object_pred_boxes, object_target_boxes)

        return f_loss + localization_loss

