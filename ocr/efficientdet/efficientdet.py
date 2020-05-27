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


if __name__ == '__main__':
    # print(EfficientNet.from_pretrained('efficientnet-b0')(torch.from_numpy(np.ones((1, 3, 128, 128))).float())[2].shape)
    # model = nn.Sequential(*EfficientNet.from_pretrained('efficientnet-b0').get_list_features())
    # print(model(torch.from_numpy(np.ones((1, 3, 128, 128))).float()))
    model = EfficientDet(2)
    a = model((torch.from_numpy(np.ones((1, 3, 1280, 1024))).float(),
               torch.from_numpy(np.array([[[401, 550, 415, 570, 0], [201, 800, 230, 820, 1]]])).float()))
    print(a[0].shape)
