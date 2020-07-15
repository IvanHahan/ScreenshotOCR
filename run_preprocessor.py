import argparse

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt

# from utils.transform import Preprocessor
from utils.image_processing import resize_image, pad_image

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='data/auto_letters/images/-11_issues.png')
parser.add_argument('--output', default='output.jpg')
parser.add_argument('--weights', default='model/best-model.pth')
parser.add_argument('--num_classes', default=1)
args = parser.parse_args()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


if __name__ == '__main__':
    # model = UNet(1, 1, False)
    # model.load_state_dict(torch.load(args.weights))
    # model.eval()
    # model = model.to('cuda')
    # x = torch.randn(1, 1, 224, 224, requires_grad=True).to('cuda')
    # torch.onnx.export(model,  # model being run
    #                   x,  # model input (or a tuple for multiple inputs)
    #                   "model.onnx",  # where to save the model (can be a file or file-like object)
    #                   export_params=True,
    #                   opset_version=12)
    model = cv2.dnn.readNetFromONNX('model.onnx')

    image = cv2.imread(args.input, 0)

    image = resize_image(image, 1408)
    image = pad_image(image, [1408, 1024])[0].astype('float32') / 255.
    image = np.expand_dims(image, -1)
    blob = cv2.dnn.blobFromImage(image)
    model.setInput(blob)
    out = model.forward().squeeze()
    out = sigmoid(out)
    plt.imshow(out)
    plt.show()

    out = (out * 255.).astype('uint8')
    cv2.imwrite('out.jpg', out)
