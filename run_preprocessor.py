import cv2
from utils.transform import Preprocessor
from unet import UNet
import argparse
import torch
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='data/manual_letters/images/15_OBFOriginal.png')
parser.add_argument('--output', default='output.jpg')
parser.add_argument('--weights', default='best-model.pth')
parser.add_argument('--num_classes', default=1)
args = parser.parse_args()

if __name__ == '__main__':

    model = UNet(1, 1).to('cuda')
    model.load_state_dict(torch.load(args.weights))
    model.eval()

    image = cv2.imread(args.input)
    with torch.no_grad():
        preprocessor = Preprocessor(False)
        image = preprocessor([image])[0]
        image = image.view(1, *image.size()).to('cuda')
        out = model(image.float())[0].cpu().numpy().transpose([1, 2, 0])[..., 0]
        plt.imshow(out)
        plt.show()
