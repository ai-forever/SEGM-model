import torch
import torchvision
import cv2
import math
import random
import numpy as np


class Normalize:
    def __call__(self, img):
        img = img.astype(np.float32) / 255
        return img


class ToTensor:
    def __call__(self, arr):
        arr = torch.from_numpy(arr)
        return arr


class MoveChannels:
    """Move the channel axis to the zero position as required in pytorch."""

    def __init__(self, to_channels_first=True):
        self.to_channels_first = to_channels_first

    def __call__(self, image):
        if self.to_channels_first:
            return np.moveaxis(image, -1, 0)
        else:
            return np.moveaxis(image, 0, -1)


class RandomGaussianBlur:
    """Apply Gaussian blur with random kernel size
    Args:
        max_ksize (int): maximal size of a kernel to apply, should be odd
        sigma_x (int): Standard deviation
    """

    def __init__(self, max_ksize=5, sigma_x=20):
        assert max_ksize % 2 == 1, "max_ksize should be odd"
        self.max_ksize = max_ksize // 2 + 1
        self.sigma_x = sigma_x

    def __call__(self, image, mask1, mask2):
        kernal_size = tuple(2 * np.random.randint(0, self.max_ksize, 2) + 1)
        blured_image = cv2.GaussianBlur(image, kernal_size, self.sigma_x)
        return blured_image, mask1, mask2


class UseWithProb:
    def __init__(self, transform, prob=0.5):
        self.transform = transform
        self.prob = prob

    def __call__(self, image, mask1, mask2):
        if random.random() < self.prob:
            image, mask1, mask2 = self.transform(image, mask1, mask2)
        return image, mask1, mask2


class ToDType:
    def __init__(self, dtype=np.float32):
        self.dtype = dtype

    def __call__(self, arr):
        return arr.astype(dtype=self.dtype)


class ExpandDims:
    def __call__(self, image):
        return np.expand_dims(image, 2)


class Flip(object):
    def __init__(self, flip_code):
        assert flip_code == 0 or flip_code == 1
        self.flip_code = flip_code

    def __call__(self, img):
        img = cv2.flip(img, self.flip_code)
        return img


class HorizontalFlip(Flip):
    def __init__(self):
        super().__init__(1)


class VerticalFlip(Flip):
    def __init__(self):
        super().__init__(0)


class Transpose:
    def __call__(self, img):
        return cv2.transpose(img)


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask1, mask2):
        for transform in self.transforms:
            image, mask1, mask2 = transform(image, mask1, mask2)
        return image, mask1, mask2


class RandomTransposeAndFlip:
    """Rotate image by randomly apply transpose, vertical or horizontal flips.
    """
    def __init__(self):
        self.transpose = Transpose()
        self.vertical_flip = VerticalFlip()
        self.horizontal_flip = HorizontalFlip()

    def __call__(self, img, mask1, mask2):
        if random.random() < 0.5:
            img = self.transpose(img)
            mask1 = self.transpose(mask1)
            mask2 = self.transpose(mask2)
        if random.random() < 0.5:
            img = self.vertical_flip(img)
            mask1 = self.vertical_flip(mask1)
            mask2 = self.vertical_flip(mask2)
        if random.random() < 0.5:
            img = self.horizontal_flip(img)
            mask1 = self.horizontal_flip(mask1)
            mask2 = self.horizontal_flip(mask2)
        return img, mask1, mask2


class Scale:
    def __init__(self, height, width):
        self.size = (width, height)

    def __call__(self, img, mask1=None, mask2=None):
        resize_img = cv2.resize(img, self.size, cv2.INTER_LINEAR)
        if mask1 is not None and mask2 is not None:
            resize_mask1 = cv2.resize(mask1, self.size, cv2.INTER_LINEAR)
            resize_mask2 = cv2.resize(mask2, self.size, cv2.INTER_LINEAR)
            return resize_img, resize_mask1, resize_mask2
        return resize_img


def img_crop(img, box):
    return img[box[1]:box[3], box[0]:box[2]]


def random_crop(img, size):
    tw = size[0]
    th = size[1]
    h, w = img.shape[:2]
    if ((w - tw) > 0) and ((h - th) > 0):
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
    else:
        x1 = 0
        y1 = 0
    img_return = img_crop(img, (x1, y1, x1 + tw, y1 + th))
    return img_return, x1, y1


class RandomCrop:
    def __init__(self, rnd_crop_min, rnd_crop_max=1):
        self.factor_max = rnd_crop_max
        self.factor_min = rnd_crop_min

    def __call__(self, img, mask1, mask2):
        factor = random.uniform(self.factor_min, self.factor_max)
        size = (
            int(img.shape[1]*factor),
            int(img.shape[0]*factor)
        )
        img, x1, y1 = random_crop(img, size)
        mask1 = img_crop(mask1, (x1, y1, x1 + size[0], y1 + size[1]))
        mask2 = img_crop(mask2, (x1, y1, x1 + size[0], y1 + size[1]))
        return img, mask1, mask2


def largest_rotated_rect(w, h, angle):
    """
    https://stackoverflow.com/a/16770343

    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell
    """

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )


def crop_around_center(image, width, height):
    """
    https://stackoverflow.com/a/16770343

    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point
    """

    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if(width > image_size[0]):
        width = image_size[0]

    if(height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]


class RandomRotate:
    """Random image rotate around the image center

    Args:
        max_ang (float): Max angle of rotation in deg
    """

    def __init__(self, max_ang=0):
        self.max_ang = max_ang

    def __call__(self, img, mask1, mask2):
        h, w, _ = img.shape

        ang = np.random.uniform(-self.max_ang, self.max_ang)
        M = cv2.getRotationMatrix2D((w/2, h/2), ang, 1)
        img = cv2.warpAffine(img, M, (w, h))

        w_cropped, h_cropped = largest_rotated_rect(w, h, math.radians(ang))
        img = crop_around_center(img, w_cropped, h_cropped)

        mask1 = cv2.warpAffine(mask1, M, (w, h))
        mask1 = crop_around_center(mask1, w_cropped, h_cropped)
        mask2 = cv2.warpAffine(mask2, M, (w, h))
        mask2 = crop_around_center(mask2, w_cropped, h_cropped)
        return img, mask1, mask2


class InferenceTransform:
    def __init__(self, height, width):
        self.transforms = torchvision.transforms.Compose([
            Scale(height, width),
            MoveChannels(to_channels_first=True),
            Normalize(),
            ToTensor()
        ])

    def __call__(self, images):
        transformed_images = []
        for image in images:
            image = self.transforms(image)
            transformed_images.append(image)
        transformed_tensor = torch.stack(transformed_images, 0)
        return transformed_tensor


def get_train_transforms(height, width, prob=0.2):
    transforms = Compose([
        UseWithProb(RandomTransposeAndFlip(), prob),
        UseWithProb(RandomCrop(rnd_crop_min=0.7, rnd_crop_max=0.95), prob),
        UseWithProb(RandomGaussianBlur(), prob),
        UseWithProb(RandomRotate(45), prob),
        Scale(height, width)
    ])
    return transforms


def get_image_transforms():
    transforms = torchvision.transforms.Compose([
        MoveChannels(to_channels_first=True),
        Normalize(),
        ToTensor()
    ])
    return transforms


def get_mask_transforms():
    transforms = torchvision.transforms.Compose([
        ExpandDims(),
        MoveChannels(to_channels_first=True),
        ToDType(dtype=np.float32),
        ToTensor()
    ])
    return transforms
