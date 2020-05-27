import torch
import torch.nn as nn
import math
from ocr.efficientnet.model import EfficientNet
from ocr.efficientdet.bifpn import BIFPN
from ocr.efficientdet.retinahead import RetinaHead
from ocr.efficientdet.module import RegressionModel, ClassificationModel, Anchors, ClipBoxes, BBoxTransform
from torchvision.ops import nms
from ocr.efficientdet.utils import calc_iou
import numpy as np
MODEL_MAP = {
    'efficientdet-d0': 'efficientnet-b0',
    'efficientdet-d1': 'efficientnet-b1',
    'efficientdet-d2': 'efficientnet-b2',
    'efficientdet-d3': 'efficientnet-b3',
    'efficientdet-d4': 'efficientnet-b4',
    'efficientdet-d5': 'efficientnet-b5',
    'efficientdet-d6': 'efficientnet-b6',
    'efficientdet-d7': 'efficientnet-b6',
}


class EfficientDet(nn.Module):
    def __init__(self,
                 num_classes,
                 network='efficientdet-d0',
                 D_bifpn=3,
                 W_bifpn=88,
                 D_class=3,
                 is_training=True,
                 threshold=0.01,
                 iou_threshold=0.5):
        super(EfficientDet, self).__init__()
        self.backbone = EfficientNet.from_pretrained(MODEL_MAP[network])
        self.is_training = is_training
        self.neck = BIFPN(in_channels=self.backbone.get_list_features()[-5:],
                          out_channels=W_bifpn,
                          stack=D_bifpn,
                          num_outs=5)
        self.bbox_head = RetinaHead(num_classes=num_classes,
                                    in_channels=W_bifpn)

        self.threshold = threshold
        self.iou_threshold = iou_threshold
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.freeze_bn()

    def forward(self, inputs):
        x = self.extract_feat(inputs)
        classes, boxes = self.bbox_head(x)
        return classes, boxes

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def extract_feat(self, img):
        """
            Directly extract features from the backbone+neck
        """
        x = self.backbone(img)
        x = self.neck(x[-5:])
        return x

    def build_label(self, annots, img_shape, anchor_ratios, num_classes):
        rect_levels = []
        classes_levels = []
        cell_shapes = []
        divider = 8

        img_h, img_w = img_shape
        for i in range(5):
            level_shape = img_h // divider, img_w // divider
            rect_level = torch.zeros((len(anchor_ratios) * 4, *level_shape))
            class_level = torch.zeros((len(anchor_ratios) * num_classes, *level_shape))
            level_cell_shapes = []
            for ratio in anchor_ratios:
                cell_size = max(img_shape[0] / level_shape[0], img_shape[1] / level_shape[1])
                cell_shape = [cell_size, cell_size]
                if ratio < 1:
                    cell_shape[0] /= ratio
                else:
                    cell_shape[1] *= ratio
                level_cell_shapes.append(cell_shape)
            cell_shapes.append(level_cell_shapes)
            rect_levels.append(rect_level)
            classes_levels.append(class_level)
            divider *= 2
        cell_shapes = torch.FloatTensor(cell_shapes)
        for annot in annots:
            x1, y1, x2, y2, c = annot
            c_x, c_y = (x2 + x1) / 2, (y2 + y1) / 2
            w, h = (x2 - x1), (y2 - y1)
            max_iou_i = torch.argmax(torch.FloatTensor([calc_iou([x1, y1, x2, y2], [0, 0, *cell_shape], no_positions=True)
                                       for level_cell_shapes in cell_shapes
                                       for cell_shape in level_cell_shapes]))
            best_level = max_iou_i // len(anchor_ratios)
            best_anchor = max_iou_i % len(anchor_ratios)

            anchor_shape = cell_shapes[best_level, best_anchor]
            cell_shape = img_h // rect_levels[best_level].shape[1], img_w // rect_levels[best_level].shape[2]
            y_i = c_y.int() // cell_shape[0]
            x_i = c_x.int() // cell_shape[1]

            rect_levels[best_level][best_anchor * 4, y_i, x_i] = (c_x - x_i * cell_shape[1]) / cell_shape[1]
            rect_levels[best_level][best_anchor * 4 + 1, y_i, x_i] = (c_y - y_i * cell_shape[0]) / cell_shape[1]
            rect_levels[best_level][best_anchor * 4 + 2, y_i, x_i] = np.log(w / anchor_shape[1])
            rect_levels[best_level][best_anchor * 4 + 3, y_i, x_i] = np.log(h / anchor_shape[0])
            classes_levels[best_level][best_anchor * num_classes + c.int(), y_i, x_i] = 1
        rects = torch.cat([l.view(-1, 4) for l in rect_levels], 0)
        classes = torch.cat([c.view(-1, num_classes) for c in classes_levels], 0)
        return rects, classes


if __name__ == '__main__':
    # print(EfficientNet.from_pretrained('efficientnet-b0')(torch.from_numpy(np.ones((1, 3, 128, 128))).float())[2].shape)
    # model = nn.Sequential(*EfficientNet.from_pretrained('efficientnet-b0').get_list_features())
    # print(model(torch.from_numpy(np.ones((1, 3, 128, 128))).float()))
    model = EfficientDet(2)
    a = model((torch.from_numpy(np.ones((1, 3, 1280, 1024))).float(),
               torch.from_numpy(np.array([[[401, 550, 415, 570, 0], [201, 800, 230, 820, 1]]])).float()))
    print(a[0].shape)
