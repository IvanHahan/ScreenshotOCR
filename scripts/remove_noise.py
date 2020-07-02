import argparse

import cv2
import numpy as np
from tqdm import tqdm

from utils.annotations import parse_annotation
from utils.image_processing import dark_on_light

parser = argparse.ArgumentParser()
parser.add_argument('--image_path', default='data/auto_letters/images/12_OBFmainframe2.png')
parser.add_argument('--annot_path', default='data/auto_letters/annotations/12_OBFmainframe2.npy')
parser.add_argument('--format', choices=['numpy', 'lblme'], default='numpy')
parser.add_argument('--output_path', default='data/auto_letters/processed/12_OBFmainframe2.png')
args = parser.parse_args()


def lblme_parser(path):
    annot = parse_annotation(path)
    for i, shape in enumerate(tqdm(annot['shapes'])):
        x1, y1 = shape['points'][0]
        x2, y2 = shape['points'][1]
        if x1 == x2 or y1 == y2:
            continue

        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        yield x1, y1, x2, y2


def numpy_parser(path):
    letter_boxes = np.load(path).astype('float32')
    letter_boxes[:, 2] = letter_boxes[:, 0] + letter_boxes[:, 2]
    letter_boxes[:, 3] = letter_boxes[:, 1] + letter_boxes[:, 3]
    for x1, y1, x2, y2 in letter_boxes.astype(int).tolist():
        yield x1, y1, x2, y2


if __name__ == '__main__':
    image = cv2.imread(args.image_path, 0)
    canvas = np.ones_like(image) * 255

    if args.format == 'numpy':
        gen = numpy_parser(args.annot_path)
    elif args.format == 'lblme':
        gen = lblme_parser(args.annot_path)

    for i, (x1, y1, x2, y2) in enumerate(tqdm(gen)):

        h = y2 - y1

        gap = h // 4
        x1_ = x1 - gap if (x1 - gap) > 0 else 0
        y1_ = y1 - gap if (y1 - gap) > 0 else 0
        x2_ = x2 + gap if (x2 + gap) < image.shape[1] else image.shape[1] - 1
        y2_ = y2 + gap if (y2 + gap) < image.shape[0] else image.shape[0] - 1

        letter = image[y1_:y2_, x1_:x2_].copy()

        if not dark_on_light(image[y1_:y2_, x1_:x2_]):
            letter = cv2.bitwise_not(letter)
        # letter = cv2.adaptiveThreshold(letter, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, -10)
        letter = cv2.threshold(letter, 200, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        canvas[y1_:y2_, x1_:x2_] = letter

    cv2.imwrite(args.output_path, canvas)
    cv2.imshow('im', canvas)
    cv2.waitKey(0)
