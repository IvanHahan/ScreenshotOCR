from ocr.efficientdet import EfficientDet
import torch
from torch import nn
from ocr.data.letter_dataset import LetterDataset
import argparse
import numpy as np
from ocr.efficientdet.losses import total_loss
from ocr.efficientdet.utils import build_label
from torch.optim import Adam

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
        annot = torch.FloatTensor([[64, 64, 68, 68, 0]])
        rects, classes = build_label(annot, img.shape[2:], [0.5, 1, 2], 2)
        classes_, rects_ = model(img)
        loss = calc_loss(rects_, classes_, rects.view(1, *rects.shape), classes.view(1, *classes.shape))
        print(loss.item())
        loss.backward()
        optimizer.step()


