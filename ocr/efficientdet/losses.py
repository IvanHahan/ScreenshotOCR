import torch
from torch import nn
from torch.nn import functional as F


def focal_loss(alpha, gamma):
    def loss(pred_classes, annot_classes):
        # boxes: samples x boxes x (xywh)
        # classes: samples x boxes x classes

        entropy = F.binary_cross_entropy(pred_classes.view(-1),
                                  annot_classes.view(-1), reduce=False)
        inv_entropy = torch.exp(entropy)
        f_loss = (alpha * (1 - inv_entropy) ** gamma) * entropy

        return f_loss.mean()
    return loss


def total_loss():
    def loss(pred_boxes, pred_classes, annot_boxes, annot_classes):
        object_i = torch.sum(annot_classes, dim=2) > 0

        object_pred_boxes = pred_boxes[object_i]
        object_annot_boxes = annot_boxes[object_i]

        f_loss = focal_loss(0.25, 2)(pred_classes, annot_classes)

        x_loss = F.binary_cross_entropy(object_pred_boxes[..., 0], object_annot_boxes[..., 0])  # x
        y_loss = F.binary_cross_entropy(object_pred_boxes[..., 1], object_annot_boxes[..., 1])  # y
        w_loss = F.mse_loss(object_pred_boxes[..., 2], object_annot_boxes[..., 2])
        h_loss = F.mse_loss(object_pred_boxes[..., 3], object_annot_boxes[..., 3])

        return f_loss + x_loss + y_loss + w_loss + h_loss
    return loss
