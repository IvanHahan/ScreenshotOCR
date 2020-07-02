import numpy as np
from torchvision import transforms
from torchvision.transforms import functional as F

from utils.image_processing import resize_image, pad_image


class MaxSizeResizer(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        """
        :param image: numpy image
        :param annotations: (x1, x2, y1, y2, label) unscaled
        """
        if isinstance(sample, dict):
            image = sample['image']
            if 'annotations' in sample:
                annotations = sample['annotations']
                d = max(image.shape) / self.size
                annotations[:, :4] /= d
                annotations = annotations.astype(int)
                sample['annotations'] = annotations

            image = resize_image(image, self.size)
            sample['image'] = image
        else:
            sample = resize_image(sample, self.size)

        return sample


class SquarePad(object):

    def __call__(self, sample):
        """
        :param image: numpy image
        :param annotations: (x1, x2, y1, y2, label) unscaled
        """
        if isinstance(sample, dict):
            image = sample['image']

            image = pad_image(image, (max(image.shape), max(image.shape)))[0]
            sample['image'] = image
        else:
            sample = pad_image(sample, (max(sample.shape), max(sample.shape)))[0]

        return sample


class Preprocessor(object):

    def __init__(self, max_size=1152, augment=True):
        self.augment = augment
        self.max_size = max_size

    def __call__(self, samples):
        """samples: pil images"""

        transformations = transforms.Compose([
            MaxSizeResizer(self.max_size),
            # SquarePad(),
            transforms.ToPILImage(),
        ])
        samples = [transformations(s) for s in samples]

        if self.augment:
            rotation, translation, scale, shear = transforms.RandomAffine.get_params((0, 0), (0.2, 0.2),
                                                                                     (1, 1.7), None,
                                                                                     samples[0].size)
            brightness = np.random.uniform(0.8, 1.2)
            hue = np.random.uniform(-0.5, 0.5)

            samples[0] = F.adjust_brightness(samples[0], brightness)
            samples[0] = F.adjust_hue(samples[0], hue)

            for i, s in enumerate(samples):
                s = F.affine(s, rotation, translation, scale, shear)
                samples[i] = s

        transformations = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor()
        ])
        samples = [transformations(s) for s in samples]
        return samples
