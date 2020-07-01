import cv2
import numpy as np


def dark_on_light(img):
    bgrd, text = dominant_colors(img, 2)
    return (bgrd < text)


def dominant_colors(image, k=3):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # Set flags (Just to avoid line break in the code)
    flags = cv2.KMEANS_PP_CENTERS

    # Apply KMeans
    data = np.reshape(image, (-1, 1)).astype('float32')
    ret, label, name_centroids = cv2.kmeans(data, k, None, criteria, 10, flags)
    return name_centroids.squeeze()[np.argsort(np.bincount(label.squeeze())).squeeze()]


def resize_image(image, size=600):
    width = int(size * image.shape[1] / image.shape[0] if image.shape[0] > image.shape[1] else size)
    height = int(size * image.shape[0] / image.shape[1] if image.shape[0] < image.shape[1] else size)
    return cv2.resize(image, (width, height))


def pad_image(image, size=(600, 600)):
    assert size[0] >= image.shape[1] and size[1] >= image.shape[0]
    dx = size[0] - image.shape[1]
    dy = size[1] - image.shape[0]

    value = ((0, dy), (0, dx), (0, 0)) if image.ndim == 3 else ((0, dy), (0, dx))

    return np.lib.pad(image, value, 'constant', constant_values=0), (dx, dy)


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

