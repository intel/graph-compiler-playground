import time
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from dl_bench.utils import Backend, Benchmark, TimerManager

from dl_bench.mlp import get_macs


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
    from torchvision.models import vgg16, resnet18, resnet50, resnext50_32x4d, resnext101_32x8d, densenet121, efficientnet_v2_m, mobilenet_v3_large

    name2model = {
        'vgg16': vgg16,
        'resnet18': resnet18,
        'resnet50': resnet50,
        'resnext50': resnext50_32x4d,
        'resnext101': resnext101_32x8d,
        'densenet121': densenet121,
        'efficientnet_v2m': efficientnet_v2_m,
        'mobilenet_v3l': mobilenet_v3_large,
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


IN_SHAPE = (3, 224, 224)

class CnnBenchmark(Benchmark):
    def run(self, backend: Backend, params):
        tm = TimerManager()
        
        try:
            print("Torch cpu capability:", torch.backends.cpu.get_cpu_capability())
            1 / 0
        except:
            pass


        # PARAMS
        name = params.get("name", "resnet50")
        batch_size = int(params.get("batch_size", 1024))

        # Do early stopping once we hit min_batches & min_seconds to accelerate measurement
        min_batches = 10
        min_seconds = 10
        DATASET_SIZE = max(10_240, batch_size * min_batches)

        trainloader, testloader = get_inf_loaders(
            DATASET_SIZE,
            in_shape=IN_SHAPE,
            batch_size=batch_size,
            device=backend.device_name,
        )
        net = get_cnn(name=name)
        # flops_per_sample = get_macs(net, IN_SHAPE, backend) * 2
        flops_per_sample = 2 * get_macs(net, IN_SHAPE, backend)

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

        diffs = []
        net.eval()
        with torch.no_grad():
            start = time.perf_counter()
            # Duration is inconsistent now
            with tm.timeit("duration_s"):
                for x in testloader:
                    s = time.perf_counter()
                    x = backend.to_device(x)
                    if backend.dtype == torch.float32:
                        output = net(x)
                        diff = time.perf_counter() - s
                        assert output.dtype is backend.dtype, f"{output.dtype}!={backend.dtype}"
                        _, predicted = torch.max(output.data, 1)

                    else:
                        with torch.autocast(device_type=backend.device_name, dtype=backend.dtype):
                            output = net(x)
                            diff = time.perf_counter() - s
                            assert output.dtype is backend.dtype, f"{output.dtype}!={backend.dtype}"
                            _, predicted = torch.max(output.data, 1)

                    diffs.append(diff)
                    n_items += len(x)

                    # early stopping
                    if (time.perf_counter() - start) > min_seconds and n_items > batch_size * min_batches:
                        break

        print(f"Latency 0%-5%-50%-95%-100% are: {np.percentile(diffs, [0, 5, 50, 95, 100])}")

        results = tm.get_results()
        results["samples_per_s"] = n_items / sum(diffs)
        print(f"Items per second: {results['samples_per_s']:.3}")
        results["flops_per_sample"] = flops_per_sample

        return results

        # print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
