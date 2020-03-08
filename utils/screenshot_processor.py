import cv2
import os


def select_roi(image, name, action):
    r = cv2.selectROI(name, image)
    roi = image[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
    action(roi, name)


class ScreenshotProcessor:

    def __init__(self, images_dir, window_size=(1000, 600)):
        self.images_dir = images_dir
        self.actions = {}
        self.window_size = window_size
        self.mouse_click = None
        self.current_screenshot_name = None
        self.current_screenshot_copy = None
        self.current_screenshot = None

    def set_mouse_callback(self, action):

        def callback(event, x, y, flags, param):
            action(self.current_screenshot, event, (x, y))
        self.mouse_click = callback
        return self

    def add_event(self, key, action):
        self.actions[ord(key)] = action
        return self

    def preprocess_image(self, screenshot):
        pass

    def reset(self):
        self.current_screenshot_copy = self.current_screenshot.copy()

    def start(self, seq_len=1):
        files = os.listdir(self.images_dir)
        files = list(filter(lambda f: os.path.splitext(f)[1].lower() in ['.png', '.jpg'], files))
        files = sorted(files, key=lambda s: int(s.split('_')[0]))
        finish = False
        for i, file in enumerate(files):
            image_name = os.path.splitext(file)[0]

            image_path = os.path.join(self.images_dir, file)

            screenshot_name = image_name
            self.current_screenshot_name = screenshot_name
            self.current_screenshot = cv2.imread(image_path)
            self.current_screenshot_copy = self.current_screenshot.copy()
            self.current_screenshot_copy = self.preprocess_image(self.current_screenshot_copy)

            cv2.namedWindow(screenshot_name, cv2.WINDOW_NORMAL)
            cv2.moveWindow(screenshot_name, 50, 50)
            cv2.resizeWindow(screenshot_name, self.window_size[0], self.window_size[1])

            if self.mouse_click is not None:
                cv2.setMouseCallback(screenshot_name, self.mouse_click)

            while True:
                cv2.imshow(screenshot_name, self.current_screenshot_copy)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("r"):
                    self.reset()

                elif key == ord("c"):
                    cv2.destroyWindow(screenshot_name)
                    break

                elif key == ord("f"):
                    finish = True
                    break

                if key in self.actions:
                    self.actions[key](self.current_screenshot_copy, image_name)

            if finish:
                break