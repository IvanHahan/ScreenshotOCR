import cv2
import numpy as np
from utils.visualization import show


def extract_lines(canny, hor_line_thresh=42, ver_line_thresh=30):

    hor_lines = cv2.morphologyEx(canny, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,
                                                                                  (hor_line_thresh, 1)))

    ver_lines = cv2.morphologyEx(canny, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,
                                                                                  (1, ver_line_thresh)))

    return cv2.bitwise_or(hor_lines, ver_lines)


def letter_boxes(image):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to grayscale
    if image.shape[0] * image.shape[1] == 0:
        return []

    h, w = image.shape[:2]

    image = cv2.pyrUp(image)

    nh, nw = image.shape

    canny = cv2.Canny(image, 50, 50)
    # show(canny)

    original = canny.copy()

    lines = extract_lines(canny, 42, 30)
    # show(lines)

    extracted = cv2.bitwise_and(original, cv2.bitwise_not(lines))
    contours = cv2.findContours(extracted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    # show(extracted)

    boxes = np.array([cv2.boundingRect(c) for c in contours])
    boxes //= 2

    keep_boxes = (boxes[:, -1] < 30) & (boxes[:, -1] > 5)
    boxes = boxes[keep_boxes]

    return boxes

