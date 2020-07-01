import cv2
import argparse
import numpy as np
from utils.annotations import parse_annotation
from utils.image_processing import dark_on_light
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--image_path', default='data/manual_letters/images/-11_issues.png')
parser.add_argument('--annot_path', default='data/manual_letters/annotations/-11_issues.json')
parser.add_argument('--output_path', default='data/manual_letters/images/-11_issues_out.png')
args = parser.parse_args()

if __name__ == '__main__':
    image = cv2.imread(args.image_path, 0)
    canvas = np.ones_like(image) * 255

    annot = parse_annotation(args.annot_path)

    for i, shape in enumerate(tqdm(annot['shapes'])):
        x1, y1 = shape['points'][0]
        x2, y2 = shape['points'][1]
        if x1 == x2 or y1 == y2:
            continue

        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        h = y2 - y1

        letter = image[y1:y2, x1:x2].copy()

        gap = h // 3
        x1_ = x1 - gap if (x1 - gap) > 0 else 0
        y1_ = y1 - gap if (y1 - gap) > 0 else 0
        x2_ = x2 + gap if (x2 + gap) < image.shape[1] else image.shape[1] - 1
        y2_ = y2 + gap if (y2 + gap) < image.shape[0] else image.shape[0] - 1

        if not dark_on_light(image[y1_:y2_, x1_:x2_]):
            letter = cv2.bitwise_not(letter)
        letter = cv2.threshold(letter, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        canvas[y1:y2, x1:x2] = letter

    cv2.imwrite(args.output_path, canvas)
    cv2.imshow('im', canvas)
    cv2.waitKey(0)


