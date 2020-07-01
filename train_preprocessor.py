import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F

from ranger import ranger
from unet import UNet
from utils.image_processing import resize_image, pad_image

device = 'cuda'

class TextDataset(Dataset):

    def __init__(self, image_dir, out_dir, transform=Preprocessor(True)):
        self.images = np.array(glob.glob(os.path.join(image_dir, '*')))
        self.outs = np.array(glob.glob(os.path.join(out_dir, '*')))
        self.transform = transform

    def __getitem__(self, item):
        image = cv2.imread(self.images[item])
        out = cv2.imread(self.outs[item])
        out = cv2.bitwise_not(out)
        image, out = self.transform(image, out)
        return image, out

    def __len__(self):
        return len(self.images)


train_dataset = TextDataset('data/manual_letters/images', 'data/manual_letters/processed')

model = UNet(1, 1).to(device)
model.train()

calc_loss = torch.nn.BCEWithLogitsLoss().to(device)

optimizer = ranger(model.parameters(), 1e-3)

min_loss = 10000

for e in range(0, 300):
    losses = []
    for images, targets in DataLoader(train_dataset, 1, False):
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()

        outs = model(images.float())

        loss = calc_loss(targets, outs)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()

    train_loss = np.mean(losses)
    if train_loss < min_loss:
        min_loss = train_loss
        torch.save(model.state_dict(), 'best-model.pth')
    print(train_loss)
    if e > 0 and e % 10 == 0:

        im = torch.sigmoid(outs[0]).detach().cpu().numpy().transpose([1, 2, 0])[..., 0]
        plt.imshow(im)
        plt.show()

    if e == 200:
        optimizer = ranger(model.parameters(), 1e-4)
