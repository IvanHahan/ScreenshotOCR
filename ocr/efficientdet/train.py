from ocr.efficientdet import EfficientDet
import torch
from torch import nn
from ocr.data.letter_dataset import LetterDataset
import argparse
import numpy as np
from ocr.efficientdet.losses import total_loss
from ocr.efficientdet.utils import build_label, postprocess
from torch.optim import Adam
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
args = parser.parse_args()

if __name__ == '__main__':

    calc_loss = total_loss()

    model = EfficientDet(2)
    optimizer = Adam(model.parameters())

    for e in range(args.epochs):
        optimizer.zero_grad()
        img = torch.zeros((1, 3, 128, 128))
        annot = torch.FloatTensor([[64, 64, 80, 80, 0]])
        rects, classes = build_label(annot, img.shape[2:], [0.5, 1, 2], 2)
        classes_, train_rects_, output_rects_ = model(img)

        out_classes, out_rects = postprocess(classes_, output_rects_)
        img = np.zeros((128, 128))
        for rect in np.unique(rects.numpy().astype(int), axis=0):
            x1, y1, x2, y2 = rect
            img[y1:y2, x1:x2] = 1
        plt.imshow(img)
        plt.show()

        loss = calc_loss(train_rects_, classes_, rects.view(1, *rects.shape), classes.view(1, *classes.shape))
        print(loss.item())
        loss.backward()
        optimizer.step()


