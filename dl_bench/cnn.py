import time
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18, resnet50, ResNet50_Weights

from dl_bench.utils import Backend, Benchmark, TimerManager

from dl_bench.mlp import RandomClsDataset, get_random_loaders, get_cifar_loaders, get_macs


def get_time():
    return time.perf_counter()


# PARAMS


def get_cnn(name):
    name2model = {
        'resnet18': resnet18,
        'resnet50': resnet50,
    }
    if name in name2model:
        return name2model[name]()
    else:
        raise ValueError(f"Unknown name {name}")


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


IN_SHAPE = (3, 250, 250)
N_CLASSES = 1000

class CnnBenchmark(Benchmark):
    def run(self, backend: Backend, params):
        tm = TimerManager()

        # PARAMS
        name = params.get("name", "resnet50")
        batch_size = int(params.get("batch_size", 1024))

        # Do early stopping once we hit min_batches & min_seconds to accelerate measurement
        min_batches = 3
        min_seconds = 10
        DATASET_SIZE = max(10_240, batch_size * min_batches)

        trainloader, testloader = get_random_loaders(
            DATASET_SIZE,
            in_shape=IN_SHAPE,
            n_classes=N_CLASSES,
            batch_size=batch_size,
            device=backend.device_name,
        )
        net = get_cnn(name=name)
        flops_per_sample = get_macs(net, IN_SHAPE, backend) * 2

        sample = backend.to_device(torch.rand(batch_size, *IN_SHAPE))
        net = backend.prepare_eval_model(net, sample_input=sample)
        print("Warmup started")
        with torch.no_grad():
            net.eval()
            with tm.timeit("warmup_s"):
                net(sample)
        print("Warmup done")

        correct = 0
        total = 0
        n_items = 0

        net.eval()
        with torch.no_grad():
            start = time.perf_counter()
            with tm.timeit("duration_s"):
                for x, y in testloader:
                    x = backend.to_device(x)
                    y = backend.to_device(y)
                    if backend.dtype == torch.float32:
                        output = net(x)
                        assert output.dtype is backend.dtype, f"{output.dtype}!={backend.dtype}"
                        _, predicted = torch.max(output.data, 1)

                        total += y.size(0)
                        correct += (predicted == y).sum().item()
                    else:
                        with torch.autocast(device_type=backend.device_name, dtype=backend.dtype):
                            output = net(x)
                            assert output.dtype is backend.dtype, f"{output.dtype}!={backend.dtype}"
                            _, predicted = torch.max(output.data, 1)

                            total += y.size(0)
                            correct += (predicted == y).sum().item()

                    n_items += len(x)

                    # early stopping
                    if (time.perf_counter() - start) > min_seconds and n_items > batch_size * min_batches:
                        break

        print(f"{n_items} were processed in {tm.name2time['duration_s']}s")

        results = tm.get_results()
        results["samples_per_s"] = n_items / results["duration_s"]
        results["flops_per_sample"] = flops_per_sample

        return results

        # print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
