from typing import Tuple
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

from torchaudio import transforms


class WineDataset(Dataset):

    def __init__(self, transform=None) -> None:
        xy = np.loadtxt('../data/wine/wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.n_samples = xy.shape[0]

        # note that we do not convert to tensor here
        self.x = xy[:, 1:]
        self.y = xy[:, [0]]

        self.transform = transform

    def __getitem__(self, index):
        sample = self.x[index], self.y[index]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.n_samples


class ToTensor:

    def __call__(self, sample) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)


class MulTransform:

    def __init__(self, factor) -> None:
        self.factor = factor

    def __call__(self, sample) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs, target = sample
        inputs *= self.factor
        return inputs, target


_dataset = WineDataset(transform=ToTensor())
first_data = _dataset[0]
features, labels = first_data
print(features)
print(type(features), type(labels))


composed = torchvision.transforms.Compose([ToTensor(), MulTransform(2)])
_dataset = WineDataset(transform=composed)
first_data = _dataset[0]
features, labels = first_data
print(features)
print(type(features), type(labels))
