import time
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from dl_bench.utils import ConcreteBenchmark


def get_time():
    return time.perf_counter()


# PARAMS
class RandomInfDataset(Dataset):
    def __init__(self, n, in_shape):
        super().__init__()

        self.values = np.random.randn(n, *in_shape).astype(np.float32)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index]


def get_inf_loaders(n, in_shape, batch_size, device: str):
    # This speeds up data copy for cuda devices
    pin_memory = device == "cuda"

    ds = RandomInfDataset(n, in_shape)
    train_loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=pin_memory
    )
    return train_loader, test_loader


def get_cnn(name):
    from torchvision.models import (
        vgg16,
        resnet18,
        resnet50,
        resnext50_32x4d,
        resnext101_32x8d,
        densenet121,
        efficientnet_v2_m,
        mobilenet_v3_large,
    )

    name2model = {
        "vgg16": vgg16,
        "resnet18": resnet18,
        "resnet50": resnet50,
        "resnext50": resnext50_32x4d,
        "resnext101": resnext101_32x8d,
        "densenet121": densenet121,
        "efficientnet_v2m": efficientnet_v2_m,
        "mobilenet_v3_large": mobilenet_v3_large,
    }
    if name in name2model:
        return name2model[name]()
    else:
        raise ValueError(f"Unknown name {name}")


class CnnBenchmark(ConcreteBenchmark):
    def __init__(self, params) -> None:
        super().__init__()

        # PARAMS
        self.in_shape = (3, 224, 224)

        name = params.get("name", "resnet50")
        self.batch_size = int(params.get("batch_size", 1024))

        # Do early stopping once we hit min_batches & min_seconds to accelerate measurement
        self.min_batches = 10
        self.min_seconds = 10

        DATASET_SIZE = max(10_240, self.batch_size * self.min_batches)

        self.dataset = RandomInfDataset(DATASET_SIZE, self.in_shape)

        self.net = get_cnn(name=name)
