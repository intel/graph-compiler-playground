import time
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from dl_bench.utils import Backend, Benchmark, TimerManager


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
IN_FEAT = 128
N_CLASSES = 10

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

name2bs = {

}


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

def get_macs(model, in_shape, backend):
    """Calculate MACs, conventional FLOPS = MACs * 2."""
    from thop import profile

    sample = torch.rand(1, *in_shape)

    model.eval()
    with torch.no_grad():
        macs, params = profile(model, inputs=(sample,), report_missing=True)

    return macs


class MlpBenchmark(Benchmark):
    def run(self, backend: Backend, params):
        tm = TimerManager()

        # PARAMS
        name = params.get("name", "size5")
        batch_size = int(params.get("batch_size", 1024))

        # Do early stopping once we hit min_batches & min_seconds to accelerate measurement
        min_batches = 10
        min_seconds = 10
        DATASET_SIZE = max(10_240, batch_size * min_batches)

        # trainloader, testloader = get_random_loaders(
        #     DATASET_SIZE,
        #     in_shape=(IN_FEAT,),
        #     n_classes=N_CLASSES,
        #     batch_size=batch_size,
        #     device=backend.device_name,
        # )
        input_tensor = torch.rand(batch_size,IN_FEAT)
        net = get_mlp(n_chans_in=IN_FEAT, n_chans_out=N_CLASSES, name=name)
        flops_per_sample = get_macs(net, (IN_FEAT,), backend) * 2

        sample = backend.to_device(torch.rand(batch_size, IN_FEAT))
        net = backend.prepare_eval_model(net, sample_input=sample)
        print("Warmup started")
        with torch.no_grad():
            net.eval()
            with tm.timeit("warmup_s"):
                net(sample)
                net(sample)
                net(sample)
        print("Warmup done")

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

        correct = 0
        total = 0
        n_items = 0

        net_time = 0

        with torch.no_grad():
            net.eval()
            start = time.perf_counter()
            with tm.timeit("duration_s"):
                # for x, y in testloader:
                #     x = backend.to_device(x)
                #     y = backend.to_device(y)
                x = input_tensor
                while True:
                    if backend.dtype == torch.float32 or backend.dtype == torch.qint8:
                        output = net(x)
                        # assert output.dtype is backend.dtype, f"{output.dtype}!={backend.dtype}"
                        _, predicted = torch.max(output.data, 1)

                        # total += y.size(0)
                        # correct += (predicted == y).sum().item()
                        total += batch_size
                    else:
                        # with torch.autocast(device_type=backend.device_name, dtype=backend.dtype):
                        tic = time.time()
                        output = net(x)
                        toc = time.time()
                        net_time += (toc - tic)
                        assert output.dtype is backend.dtype, f"{output.dtype}!={backend.dtype}"
                        _, predicted = torch.max(output.data, 1)

                        # total += y.size(0)
                        # correct += (predicted == y).sum().item()
                        total += batch_size

                    n_items += len(x)

                    # early stopping
                    if (time.perf_counter() - start) > min_seconds and total > batch_size * min_batches:
                        break

        print("net time: ", net_time)
        print(f"{n_items} were processed in {tm.name2time['duration_s']}s")

        results = tm.get_results()
        results["samples_per_s"] = n_items / results["duration_s"]
        results["flops_per_sample"] = flops_per_sample

        return results

        # print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
