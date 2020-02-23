from ocr.data.image_processing import letter_boxes
from ocr.data.visualization import draw_boxes
import os
import argparse
from utils.path import abs_path, make_dir_if_needed
from utils.visualization import show
import cv2
import numpy as np
import json


def make_annot_dt(boxes, image_shape, maintainer='ivanhahanov@icloud.com'):
    annotations = []
    for b in boxes:
        annotation = {
            'label': ['letter'],
            'shape': ['box'],
            'points': b,
            'notes': '',
            'imageWidth': image_shape[1],
            'imageHeight': image_shape[0]
        }
        annotations.append(annotation)
    return {
        'annotations': annotations,
        'maintainer': maintainer
    }


def main(args):
    boundaries_path = os.path.join(args.output_dir, 'letter_boundaries')
    outlined_path = os.path.join(args.output_dir, 'letter_outlined')

    make_dir_if_needed(boundaries_path)
    make_dir_if_needed(outlined_path)

    f = open(args.annot_path, 'w')

    for image_file in os.listdir(args.image_dir):
        image_path = os.path.join(args.image_dir, image_file)
        image = cv2.imread(image_path, 0)
        boxes = letter_boxes(image)
        canvas = np.zeros_like(image)

        outlined = draw_boxes(image, boxes)
        boundaries = draw_boxes(canvas, boxes, colored=False)

        cv2.imwrite(os.path.join(boundaries_path, image_file), boundaries)
        cv2.imwrite(os.path.join(outlined_path, image_file), outlined)

        annotation = make_annot_dt(boxes, image.shape[:2])

        f.write(json.dumps(annotation))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', default=abs_path('data/images/raw'))
    parser.add_argument('--output_dir', default=abs_path('data/images'))
    parser.add_argument('--annot_path', default=abs_path('data/text/annotations.json'))
    main(parser.parse_args())
