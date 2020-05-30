from ocr.efficientdet import EfficientDet
import torch
from torch import nn
from ocr.data.letter_dataset import LetterDataset
import argparse
import numpy as np
from ocr.efficientdet.losses import total_loss, focal_loss
from ocr.efficientdet.utils import build_label, postprocess
from torch.optim import Adam
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=500)
args = parser.parse_args()

if __name__ == '__main__':

    calc_loss = total_loss()

    model = EfficientDet(2)
    # model.eval()
    optimizer = Adam(model.parameters(), lr=0.01)

    for e in range(args.epochs):
        optimizer.zero_grad()
        img = torch.zeros((1, 3, 128, 128))
        img[..., 32:36, 32:36] = 1
        annot = torch.FloatTensor([[7, 12, 7, 12, 0]])
        rects, classes = build_label(annot, img.shape[2:], [0.5, 1, 2], 2)
        # classes = torch.zeros((1, 1, 16, 16))
        # classes[..., 3:4, 3:4] = 1
        # classes_, train_rects_, output_rects_ = model(img)
        classes_, train_boxes, output_boxes = model(img)
        loss = focal_loss(0.25, 2)(classes_, classes)
        # out_classes, out_rects = postprocess(classes_, output_rects_)
        img = np.zeros((128, 128))
        for rect in np.unique(rects.numpy().astype(int), axis=0):
            x1, y1, x2, y2 = rect
            img[y1:y2, x1:x2] = 1
        if e % 40 == 0:
            plt.imshow(classes_[0, 0].data.numpy())
            plt.show()

        # loss = calc_loss(train_rects_, classes_, rects.view(1, *rects.shape), classes.view(1, *classes.shape))
        #     print('class idx:', classes_[..., 0].argmax(), 'annot class idx:', classes[..., 0].argmax())
            print('loss:', loss.item())

        loss.backward()
        optimizer.step()


