import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from utils import Backend, Benchmark


class RandomClsDataset(Dataset):
    def __init__(self, n, in_shape, n_classes):
        super().__init__()

        self.values = np.random.randn(n, *in_shape).astype(np.float32)
        self.labels = np.random.randint(n_classes, size=(n,))

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index], self.labels[index]


def get_random_loaders(n, in_shape, n_classes, batch_size, device: str):
    # This speeds up data copy for cuda devices
    pin_memory = device == "cuda"

    ds = RandomClsDataset(n, in_shape, n_classes)
    train_loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=pin_memory
    )
    return train_loader, test_loader


def get_cifar_loaders(train_batch_size, inf_batch_size):
    import torchvision
    import torchvision.transforms as transforms

    n_chans_in = 3072
    n_chans_out = 10

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.0, 0.0, 0.0), (1, 1, 1)),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=train_batch_size, shuffle=True, num_workers=2
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=inf_batch_size, shuffle=False, num_workers=2
    )

    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )
    return trainloader, testloader


def get_time():
    return time.perf_counter()


# PARAMS
# Total dataset size
DATASET_SIZE = 10_000
N_EPOCHS = 3


BATCH_SIZE = 1024

IN_SHAPE = (128,)
N_CLASSES = 10

size2_struct = [512, 1024, 2048, 512]
size5_struct = [1024, 4096, 8192, 16384, 8192, 1024, 1024, 256]


name2params = {
    "basic": dict(struct=[16, 16]),
    "size1": dict(struct=[256, 256]),
    "size2": dict(struct=size2_struct),
    "size3": dict(struct=[512, 1024, 4096, 2048, 512]),
    "size4": dict(struct=[1024, 4096, 8192, 4096, 1024, 256]),
    "size5": dict(struct=size5_struct),
    "size2_inplace": dict(struct=size2_struct, inplace=True),
    "size2_sigm": dict(struct=size2_struct, activ_layer=nn.Sigmoid),
    "size2_tanh": dict(struct=size2_struct, activ_layer=nn.Tanh),
    "size2_gelu": dict(struct=size2_struct, activ_layer=nn.GELU),
    "size2_bn": dict(struct=size2_struct, norm_layer=nn.BatchNorm1d),
    "size5_bn_gelu": dict(
        struct=size5_struct, norm_layer=nn.BatchNorm1d, activ_layer=nn.GELU
    ),
}


def build_mlp(
    n_chans_in: int,
    struct,
    norm_layer=None,
    activ_layer=nn.ReLU,
    inplace=None,
    dropout: float = None,
):
    params = {} if inplace is None else {"inplace": inplace}
    bias = True if dropout is None else False

    layers = []
    in_dim = n_chans_in
    for hidden_dim in struct[:-1]:
        layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))
        if norm_layer is not None:
            layers.append(norm_layer(hidden_dim))
        layers.append(activ_layer(**params))
        if dropout is not None:
            layers.append(torch.nn.Dropout(dropout, **params))
        in_dim = hidden_dim

    layers.append(torch.nn.Linear(in_dim, struct[-1], bias=True))

    return nn.Sequential(*layers)


def get_mlp(n_chans_in, name):
    params = name2params[name]

    # net = nn.Sequential(nn.Flatten(), build_mlp(n_chans_in, **params))
    net = build_mlp(n_chans_in, **params)
    return net


class MlpBenchmark(Benchmark):
    def run(self, backend: Backend, params):
        name = params.get("name", "size3")
        trainloader, testloader = get_random_loaders(
            DATASET_SIZE,
            in_shape=IN_SHAPE,
            n_classes=N_CLASSES,
            batch_size=BATCH_SIZE,
            device=backend.device_name,
        )
        net = get_mlp(n_chans_in=IN_SHAPE[0], name=name)
        sample = torch.rand(1, IN_SHAPE[0])
        net = backend.prepare_eval_model(net, sample_input=sample)

        # We are not interested in training yet.
        # criterion = nn.CrossEntropyLoss()
        # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

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

        correct = 0
        total = 0

        n_items = 0

        net.eval()
        with torch.no_grad():
            start = get_time()
            for x, y in testloader:
                outputs = net(x)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

                n_items += len(x)
            stop = get_time()

        print(f"{n_items} {stop - start}")
        total_time = stop - start
        results = {
            "time": total_time,
            "samples_per_s": n_items / total_time,
        }
        return results

        # print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
