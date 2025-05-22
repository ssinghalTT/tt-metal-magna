"""Contains image segmentation transforms."""

import random
from typing import List, Tuple

import numpy as np
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F


def pad_if_smaller(img, size, fill=0):
    target_width = size[1]
    target_height = size[0]
    width, height = F.get_image_size(img)
    img_size = (height, width)
    if img_size < size:
        padh = target_height - height if height < target_height else 0
        padw = target_width - width if width < target_width else 0
        left, top = padw // 2, padh // 2
        rigth, bottom = padw - left, padh - top
        img = F.pad(img, (left, top, rigth, bottom), fill=fill)  # left, top, right and bottom
    return img


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomResize:
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        image = F.resize(image, size)
        target = F.resize(target, size, interpolation=T.InterpolationMode.NEAREST)
        return image, target


class RandomScaledResize:
    def __init__(self, scale: Tuple[float, float]):
        self._scale = scale

    def __call__(self, image, target):
        scale = np.random.uniform(self._scale[0], self._scale[1])
        width, height = F.get_image_size(image)
        size = int(height * scale), int(width * scale)
        image = F.resize(image, size)
        target = F.resize(target, size, interpolation=T.InterpolationMode.NEAREST)
        return image, target


class RandomResizedCrop:
    def __init__(self, size: Tuple[int, int], scale: Tuple[float, float], ratio: Tuple[float, float]):
        self._size = size
        self._scale = scale
        self._ratio = ratio

    def __call__(self, image, target):
        i, j, h, w = T.RandomResizedCrop.get_params(image, self._scale, self._ratio)

        image = F.resized_crop(image, i, j, h, w, self._size, interpolation=T.InterpolationMode.BILINEAR)
        target = F.resized_crop(target, i, j, h, w, self._size, interpolation=T.InterpolationMode.NEAREST)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class RandomVerticalFlip:
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.vflip(image)
            target = F.vflip(target)
        return image, target


class RandomCrop:
    def __init__(self, size):
        self.size = tuple(size)

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size)
        crop_params = T.RandomCrop.get_params(image, self.size)
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target


class CenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target


class PILToTensor:
    def __call__(self, image, target):
        image = F.pil_to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target


class ConvertImageDtype:
    def __init__(self, dtype=torch.float):
        self.dtype = dtype

    def __call__(self, image, target):
        image = F.convert_image_dtype(image, self.dtype)
        return image, target


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target

class DeNormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image[0,:,:] = image[0,:,:]*self.std[0] + self.mean[0]
        image[1,:,:] = image[1,:,:]*self.std[1] + self.mean[1]
        image[2,:,:] = image[2,:,:]*self.std[2] + self.mean[2]
        return image, target

class RandomRotation:
    def __init__(self, degrees: List[float]) -> None:
        self._degrees = degrees

    def __call__(self, image, target):
        rot_params = T.RandomRotation.get_params(self._degrees)

        image = F.rotate(image, rot_params, interpolation=T.InterpolationMode.BILINEAR)
        target = F.rotate(target, rot_params, interpolation=T.InterpolationMode.NEAREST)
        return image, target


class ColorJitter:
    def __init__(self, brightness, contrast, saturation, hue) -> None:
        self._jitter = T.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, image, target):
        image = self._jitter(image)
        return image, target
