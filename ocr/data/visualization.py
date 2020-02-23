import cv2


def draw_boxes(image, boxes, colored=True):
    image = image.copy()
    if colored:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for b in boxes:
        cv2.rectangle(image, (b[0], b[1]), (b[0]+b[2], b[1]+b[3]), (255, 0, 0), 1)
    return image
