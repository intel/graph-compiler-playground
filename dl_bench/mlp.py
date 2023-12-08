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


class RandomClsDataset(Dataset):
    def __init__(self, n, in_shape, n_classes, seed=42):
        super().__init__()
        np.random.seed(seed)
        self.values = np.random.randn(n, *in_shape).astype(np.float32)
        # self.labels = np.random.randint(n_classes, size=(n,))

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index]  # , self.labels[index]


def get_random_loaders(n, in_shape, n_classes, batch_size, device: str):
    # This speeds up data copy for cuda devices
    pin_memory = device == "cuda"

    ds = RandomClsDataset(42, n, in_shape, n_classes)
    train_loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=pin_memory
    )
    return train_loader, test_loader


size2_struct = [512, 1024, 2048, 512]
# TODO: why 8192 to 1024, need to make more gradual, update while we recalc on cuda & amd
size5_struct = [1024, 4096, 8192, 16384, 8192, 1024, 1024, 256]


name2params = {
    "basic": dict(struct=[16, 16]),
    "size1": dict(struct=[256, 256]),
    "size2": dict(struct=size2_struct),
    "size3": dict(struct=[512, 1024, 4096, 2048, 512]),
    "size4": dict(struct=[1024, 4096, 8192, 4096, 1024, 256]),
    "size5": dict(struct=size5_struct),
    "size5_sigm": dict(struct=size5_struct, activ_layer=nn.Sigmoid),
    "size5_tanh": dict(struct=size5_struct, activ_layer=nn.Tanh),
    "size5_gelu": dict(struct=size5_struct, activ_layer=nn.GELU),
    "size5_linear": dict(struct=size5_struct, activ_layer=None),
    "size5_inplace": dict(struct=size5_struct, inplace=True),
    "size5_bn": dict(struct=size5_struct, norm_layer=nn.BatchNorm1d),
    "size5_bn_gelu": dict(
        struct=size5_struct, norm_layer=nn.BatchNorm1d, activ_layer=nn.GELU
    ),
    "size5_drop_gelu": dict(struct=size5_struct, dropout=0.5, activ_layer=nn.GELU),
    "100@512": dict(struct=[512] * 100),
    # "100@512": dict(struct=[512] * 100),
    "25@1024": dict(struct=[1024] * 25),
    "4@16384": dict(struct=[16384] * 4),
    "2@16384": dict(struct=[16384] * 2),
}

name2bs = {}


def build_mlp(
    n_chans_in: int,
    n_chans_out: int,
    struct: List[int],
    norm_layer=None,
    activ_layer=nn.ReLU,
    inplace=None,
    dropout: float = None,
):
    params = {} if inplace is None else {"inplace": inplace}
    bias = True if dropout is None else False

    layers = []
    in_dim = n_chans_in
    for hidden_dim in struct:
        layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))
        if norm_layer is not None:
            layers.append(norm_layer(hidden_dim))
        if activ_layer is not None:
            layers.append(activ_layer(**params))
        if dropout is not None:
            layers.append(torch.nn.Dropout(dropout, **params))
        in_dim = hidden_dim

    layers.append(torch.nn.Linear(in_dim, n_chans_out, bias=True))

    return nn.Sequential(*layers)


def get_mlp(n_chans_in, n_chans_out, name):
    params = name2params[name]

    # net = nn.Sequential(nn.Flatten(), build_mlp(n_chans_in, **params))
    net = build_mlp(n_chans_in, **params, n_chans_out=n_chans_out)
    return net


class MlpBenchmark(ConcreteBenchmark):
    def __init__(self, params) -> None:
        super().__init__()

        # PARAMS
        IN_FEAT = 128
        N_CLASSES = 10

        self.in_shape = (IN_FEAT,)

        # PARAMS
        name = params.get("name", "size5")
        self.batch_size = int(params.get("batch_size", 1024))

        # Do early stopping once we hit min_batches & min_seconds to accelerate measurement
        self.min_batches = 10
        self.min_seconds = 10

        DATASET_SIZE = max(10_240, self.batch_size * self.min_batches)

        self.dataset = RandomClsDataset(DATASET_SIZE, self.in_shape, N_CLASSES, 42)

        self.net = get_mlp(n_chans_in=IN_FEAT, n_chans_out=N_CLASSES, name=name)

    def train(self):
        # We are not interested in training yet.
        # criterion = nn.CrossEntropyLoss()
        # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        # N_EPOCHS = 3
        # epoch_stats = {}
        # n_report = 10
        # for epoch in range(n_epochs):  # loop over the dataset multiple times
        #     running_loss = 0.0

        #     n_items = 0
        #     start = get_time()
        #     for i, (x, y) in enumerate(trainloader):
        #         optimizer.zero_grad()

        #         outputs = net(x)
        #         loss = criterion(outputs, y)
        #         loss.backward()
        #         optimizer.step()

        #         n_items += len(x)

        #         running_loss += loss.item()
        #         if i % n_report == (n_report - 1):
        #             print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / n_report:.3f}')
        #             running_loss = 0.0

        #     stop = get_time()
        #     print(f"{n_items} took {stop - start}")

        # print('Finished Training')
        pass
