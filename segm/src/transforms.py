import torch
import torchvision
import cv2
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


class UseWithProb:
    def __init__(self, transform, prob=0.5):
        self.transform = transform
        self.prob = prob

    def __call__(self, image):
        if random.random() < self.prob:
            image = self.transform(image)
        return image


class ToDType:
    def __init__(self, dtype=np.float32):
        self.dtype = dtype

    def __call__(self, arr):
        return arr.astype(dtype=self.dtype)


class ExpandDims:
    def __call__(self, image):
        return np.expand_dims(image, 2)


def get_train_transforms():
    transforms = torchvision.transforms.Compose([
        MoveChannels(to_channels_first=True),
        Normalize(),
        ToTensor()
    ])
    return transforms


def get_val_transforms():
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
