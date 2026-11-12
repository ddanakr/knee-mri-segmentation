import torch
import random

class Compose:

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)

        return img, mask

class RandomFlip:

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            img = torch.flip(img, dims=[-1])
            mask = torch.flip(mask, dims=[-1])
        if random.random() < self.p:
            img = torch.flip(img, dims=[-2])
            mask = torch.flip(mask, dims=[-2])
        return img, mask


class RandomRotate90:

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            k = random.choice([1, 2, 3])
            img = torch.rot90(img, k=k, dims=[-2, -1])
            mask = torch.rot90(mask, k=k, dims=[-2, -1])
        return img, mask

class RandomIntensityShift:

    def __init__(self, p: float = 0.5, shift_range: float = 0.1):
        self.p = p
        self.shift_range = shift_range

    def __call__(self, img, mask):
        if random.random() < self.p:
            shift = random.uniform(-self.shift_range, self.shift_range)
            img = img + shift
        return img, mask
