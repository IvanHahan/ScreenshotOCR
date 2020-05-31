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
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=500)
args = parser.parse_args()

if __name__ == '__main__':

    calc_loss = total_loss()

    model = EfficientDet(2)

    optimizer = Adam(model.parameters(), lr=0.001)

    for e in range(args.epochs):
        optimizer.zero_grad()
        img = torch.zeros((1, 3, 128, 256))
        img[..., 7:12, 7:12] = 1
        img[..., 46:58, 56:78] = 1
        annot = torch.FloatTensor([[7, 7, 12, 12, 0], [56, 46, 78, 58, 0]])
        rects, classes = build_label(annot.clone(), img.shape[2:], [0.5, 1, 2], 2)
        # classes = torch.zeros((1, 1, 16, 16))
        # classes[..., 3:4, 3:4] = 1
        # classes_, train_rects_, output_rects_ = model(img)
        classes_, activations, train_rects, output_rects = model(img)
        # loss = focal_loss(0.25, 2)(classes_, classes)

        out_classes, out_rects = postprocess(classes_[0], output_rects[0])
        if e % 40 == 0:
            img = np.zeros((128, 256, 3), dtype='uint8')
            for rect in annot.numpy():
                x1, y1, x2, y2, _ = np.clip(rect, 0, 128)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), cv2.FILLED)
            for rect in out_rects.data.numpy().astype(int):
                x1, y1, x2, y2 = rect
                img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
            plt.imshow(img)
            plt.show()
            # for activation in activations:
            #     act = np.sum(activation[0].data.numpy(), axis=-1)
            #     act = (act / act.max()) * 255.
            #     act = act.astype('uint8')
            #     plt.imshow(act)
            #     plt.show()

            # plt.imshow(classes_[0, 0].data.numpy())
            plt.show()

        loss = calc_loss(train_rects, classes_, rects.view(1, *rects.shape), classes.view(1, *classes.shape))
        print('loss:', loss.item())
        #     print('class idx:', classes_[..., 0].argmax(), 'annot class idx:', classes[..., 0].argmax())

        loss.backward()
        optimizer.step()
        # if e > 40:
        #     optimizer = Adam(model.parameters(), 0.001)


