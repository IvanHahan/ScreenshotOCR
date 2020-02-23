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
