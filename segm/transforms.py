import torch
import torchvision
import cv2
import math
import random
import numpy as np

from albumentations import augmentations


class Normalize:
    def __call__(self, img):
        img = img.astype(np.float32) / 255
        return img


class ToTensor:
    def __call__(self, arr):
        arr = torch.from_numpy(arr)
        return arr


class MoveChannels:
    """Move the channel axis to the zero position as required in
    pytorch (NxCxHxW)."""

    def __init__(self, to_channels_first=True):
        self.to_channels_first = to_channels_first

    def __call__(self, image):
        if self.to_channels_first:
            return np.moveaxis(image, -1, 0)
        else:
            return np.moveaxis(image, 0, -1)


class UseWithProb:
    def __init__(self, transform, prob=0.5):
        self.transform = transform
        self.prob = prob

    def __call__(self, image, mask):
        if random.random() < self.prob:
            image, mask = self.transform(image, mask)
        return image, mask


class OneOf:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        return random.choice(self.transforms)(image, mask)


class ToDType:
    def __init__(self, dtype=np.float32):
        self.dtype = dtype

    def __call__(self, arr):
        return arr.astype(dtype=self.dtype)


class ExpandDimsIfNeeded:
    """Add third channel for masks with one class.
    This is needed ss most of cv2 functions tends to
    remove single channel (H, W, 1) -> (H, W)
    """
    def __call__(self, image):
        if len(image.shape) == 3:
            return image
        return np.expand_dims(image, -1)


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

    def __call__(self, image, mask):
        for transform in self.transforms:
            image, mask = transform(image, mask)
        return image, mask


class RandomTransposeAndFlip:
    """Rotate image by randomly apply transpose, vertical or horizontal flips.
    """
    def __init__(self):
        self.transpose = Transpose()
        self.vertical_flip = VerticalFlip()
        self.horizontal_flip = HorizontalFlip()

    def __call__(self, img, mask):
        if random.random() < 0.5:
            img = self.transpose(img)
            mask = self.transpose(mask)
        if random.random() < 0.5:
            img = self.vertical_flip(img)
            mask = self.vertical_flip(mask)
        if random.random() < 0.5:
            img = self.horizontal_flip(img)
            mask = self.horizontal_flip(mask)
        return img, mask


class Scale:
    def __init__(self, height, width):
        self.size = (width, height)

    def __call__(self, img, mask=None):
        resize_img = cv2.resize(img, self.size, cv2.INTER_LINEAR)
        if mask is not None:
            resize_mask = cv2.resize(mask, self.size, cv2.INTER_LINEAR)
            return resize_img, resize_mask
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

    def __call__(self, img, mask):
        factor = random.uniform(self.factor_min, self.factor_max)
        size = (
            int(img.shape[1]*factor),
            int(img.shape[0]*factor)
        )
        img, x1, y1 = random_crop(img, size)
        mask = img_crop(mask, (x1, y1, x1 + size[0], y1 + size[1]))
        return img, mask


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


class RotateAndCrop:
    """Random image rotate around the image center

    Args:
        max_ang (float): Max angle of rotation in deg
    """

    def __init__(self, max_ang=90):
        self.max_ang = max_ang

    def __call__(self, img, mask):
        h, w, _ = img.shape

        ang = np.random.uniform(-self.max_ang, self.max_ang)
        M = cv2.getRotationMatrix2D((w/2, h/2), ang, 1)
        img = cv2.warpAffine(img, M, (w, h))

        w_cropped, h_cropped = largest_rotated_rect(w, h, math.radians(ang))
        img = crop_around_center(img, w_cropped, h_cropped)

        mask = cv2.warpAffine(mask, M, (w, h))
        mask = crop_around_center(mask, w_cropped, h_cropped)
        return img, mask


class InferenceTransform:
    def __init__(self, height, width, return_numpy=False):
        self.transforms = torchvision.transforms.Compose([
            Scale(height, width),
            MoveChannels(to_channels_first=True),
            Normalize(),
        ])
        self.return_numpy = return_numpy
        self.to_tensor = ToTensor()

    def __call__(self, images):
        transformed_images = [self.transforms(image) for image in images]
        transformed_array = np.stack(transformed_images, 0)
        if not self.return_numpy:
            transformed_array = self.to_tensor(transformed_array)
        return transformed_array


class CLAHE:
    def __init__(self, prob):
        self.aug = augmentations.transforms.CLAHE(p=prob)

    def __call__(self, img, mask):
        img = self.aug(image=img)['image']
        return img, mask


class MotionBlur:
    def __init__(self, prob):
        self.aug = augmentations.transforms.MotionBlur(
            blur_limit=5, p=prob)

    def __call__(self, img, mask):
        img = self.aug(image=img)['image']
        return img, mask


class ToGray:
    def __init__(self, prob):
        self.aug = augmentations.transforms.ToGray(p=prob)

    def __call__(self, img, mask):
        img = self.aug(image=img)['image']
        return img, mask


class ToSepia:
    def __init__(self, prob):
        self.aug = augmentations.transforms.ToSepia(p=prob)

    def __call__(self, img, mask):
        img = self.aug(image=img)['image']
        return img, mask


class RandomFog:
    def __init__(self, prob):
        self.aug = augmentations.transforms.RandomFog(
            fog_coef_lower=0.05, fog_coef_upper=0.2, alpha_coef=0.5, p=prob)

    def __call__(self, img, mask):
        img = self.aug(image=img)['image']
        return img, mask


class GlassBlur:
    def __init__(self, prob):
        self.aug = augmentations.transforms.GlassBlur(
            sigma=0.7, max_delta=1, p=prob)

    def __call__(self, img, mask):
        img = self.aug(image=img)['image']
        return img, mask


class Blur:
    def __init__(self, prob):
        self.aug = augmentations.transforms.Blur(p=prob)

    def __call__(self, img, mask):
        img = self.aug(image=img)['image']
        return img, mask


class ElasticTransform:
    def __init__(self, prob):
        self.aug = augmentations.geometric.transforms.ElasticTransform(p=prob)

    def __call__(self, img, mask):
        augmented = self.aug(image=img, mask=mask)
        return augmented['image'], augmented['mask']


class GridDistortion:
    def __init__(self, prob):
        self.aug = augmentations.transforms.GridDistortion(p=prob)

    def __call__(self, img, mask):
        augmented = self.aug(image=img, mask=mask)
        return augmented['image'], augmented['mask']


class OpticalDistortion:
    def __init__(self, prob):
        self.aug = augmentations.transforms.OpticalDistortion(
            distort_limit=0.08, shift_limit=0.08, p=prob)

    def __call__(self, img, mask):
        augmented = self.aug(image=img, mask=mask)
        return augmented['image'], augmented['mask']


class Rotate:
    def __init__(self, max_ang, prob):
        self.aug = augmentations.geometric.rotate.Rotate(limit=max_ang, p=prob)

    def __call__(self, img, mask):
        augmented = self.aug(image=img, mask=mask)
        return augmented['image'], augmented['mask']


class CoarseDropout:
    def __init__(self, prob):
        self.aug = augmentations.CoarseDropout(
            max_holes=12, max_height=15, max_width=15, p=prob)

    def __call__(self, img, mask):
        augmented = self.aug(image=img, mask=mask)
        return augmented['image'], augmented['mask']


class RandomRain:
    def __init__(self, prob):
        self.aug = augmentations.transforms.RandomRain(
            blur_value=1, p=prob)

    def __call__(self, img, mask):
        augmented = self.aug(image=img, mask=mask)
        return augmented['image'], augmented['mask']


class RandomSnow:
    def __init__(self, prob):
        self.aug = augmentations.transforms.RandomSnow(
            brightness_coeff=1.5, p=prob)

    def __call__(self, img, mask):
        augmented = self.aug(image=img, mask=mask)
        return augmented['image'], augmented['mask']


class ChannelDropout:
    def __init__(self, prob):
        self.aug = augmentations.ChannelDropout(p=prob)

    def __call__(self, img, mask):
        img = self.aug(image=img)['image']
        return img, mask


class MedianBlur:
    def __init__(self, prob):
        self.aug = augmentations.transforms.MedianBlur(
            blur_limit=3, p=prob)

    def __call__(self, img, mask):
        img = self.aug(image=img)['image']
        return img, mask


class GaussNoise:
    def __init__(self, prob):
        self.aug = augmentations.transforms.GaussNoise(
            p=prob)

    def __call__(self, img, mask):
        img = self.aug(image=img)['image']
        return img, mask


class ISONoise:
    def __init__(self, prob):
        self.aug = augmentations.transforms.ISONoise(
            p=prob)

    def __call__(self, img, mask):
        img = self.aug(image=img)['image']
        return img, mask


class MultiplicativeNoise:
    def __init__(self, prob):
        self.aug = augmentations.transforms.MultiplicativeNoise(
            multiplier=(0.85, 1.15), p=prob)

    def __call__(self, img, mask):
        img = self.aug(image=img)['image']
        return img, mask


class ChannelShuffle:
    def __init__(self, prob):
        self.aug = augmentations.transforms.ChannelShuffle(p=prob)

    def __call__(self, img, mask):
        img = self.aug(image=img)['image']
        return img, mask


class Posterize:
    def __init__(self, prob):
        self.aug = augmentations.transforms.Posterize(p=prob)

    def __call__(self, img, mask):
        img = self.aug(image=img)['image']
        return img, mask


class RGBShift:
    def __init__(self, prob):
        self.aug = augmentations.transforms.RGBShift(p=prob)

    def __call__(self, img, mask):
        img = self.aug(image=img)['image']
        return img, mask


class RandomBrightnessContrast:
    def __init__(self, prob):
        self.aug = augmentations.transforms.RandomBrightnessContrast(p=prob)

    def __call__(self, img, mask):
        img = self.aug(image=img)['image']
        return img, mask


class RandomGamma:
    def __init__(self, prob):
        self.aug = augmentations.transforms.RandomGamma(p=prob)

    def __call__(self, img, mask):
        img = self.aug(image=img)['image']
        return img, mask


class HueSaturationValue:
    def __init__(self, prob):
        self.aug = augmentations.transforms.HueSaturationValue(p=prob)

    def __call__(self, img, mask):
        img = self.aug(image=img)['image']
        return img, mask


class ImageCompression:
    def __init__(self, prob):
        self.aug = augmentations.transforms.ImageCompression(
            quality_lower=60, quality_upper=90, p=prob)

    def __call__(self, img, mask):
        img = self.aug(image=img)['image']
        return img, mask


class Sharpen:
    def __init__(self, prob):
        self.aug = augmentations.Sharpen(
            p=prob)

    def __call__(self, img, mask):
        img = self.aug(image=img)['image']
        return img, mask


def get_train_transforms(height, width, prob=0.3):
    transforms = Compose([
        UseWithProb(RandomTransposeAndFlip(), 1),
        OneOf([
            CLAHE(prob),
            GaussNoise(prob),
            ISONoise(prob),
            MultiplicativeNoise(prob),
            ImageCompression(prob),
            Sharpen(prob)
        ]),
        OneOf([
            CoarseDropout(prob)
        ]),
        UseWithProb(RandomCrop(rnd_crop_min=0.75), prob),
        OneOf([
            UseWithProb(RotateAndCrop(45), prob),
            Rotate(45, prob)
        ]),
        OneOf([
            ChannelDropout(prob),
            ChannelShuffle(prob),
            Posterize(prob),
            RGBShift(prob),
            ToGray(prob),
            ToSepia(prob)
        ]),
        OneOf([
            RandomBrightnessContrast(prob),
            RandomGamma(prob),
            HueSaturationValue(prob),
        ]),
        OneOf([
            RandomRain(prob),
            MotionBlur(prob),
            RandomFog(prob),
            GlassBlur(prob),
            MedianBlur(prob),
        ]),
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
        ExpandDimsIfNeeded(),
        MoveChannels(to_channels_first=True),  # move channel axis to zero position
        ToDType(dtype=np.float32),
        ToTensor()
    ])
    return transforms
