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

