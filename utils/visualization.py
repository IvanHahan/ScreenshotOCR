import matplotlib.pyplot as plt
import cv2


def show_plt(image):
    plt.imshow(image)
    plt.show()


def show_cv(img, name='image'):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 600, 400)
    cv2.imshow(name, img)
    cv2.waitKey()
    cv2.destroyWindow(name)


def show(image, backend='plt'):
    if backend == 'cv':
        show_cv(image)
    elif backend == 'plt':
        show_plt(image)


def draw_boxes(image, boxes, colored=True):
    image = image.copy()
    if colored and image.ndim < 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for b in boxes:
        cv2.rectangle(image, (b[0], b[1]), (b[0]+b[2], b[1]+b[3]), (255, 0, 0), 1)
    return image