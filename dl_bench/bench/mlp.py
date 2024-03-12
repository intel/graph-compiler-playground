from typing import List

import torch
import torch.nn as nn

from dl_bench.benchmark import Benchmark, RandomInfDataset


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


class MlpBenchmark(Benchmark):
    def __init__(self, params) -> None:
        IN_FEAT = 128
        N_CLASSES = 10
        in_shape = (IN_FEAT,)

        batch_size = int(params.get("batch_size", 1024))

        min_batches = int(params.get("min_batches", 20))
        min_seconds = int(params.get("min_seconds", 20))
        warmup = 10
        DATASET_SIZE = max(102_400, batch_size * (min_batches + warmup))
        dataset = RandomInfDataset(DATASET_SIZE, in_shape)

        name = params.get("name", "size5")
        net = get_mlp(n_chans_in=IN_FEAT, n_chans_out=N_CLASSES, name=name)

        super().__init__(
            net=net,
            in_shape=in_shape,
            dataset=dataset,
            batch_size=batch_size,
            min_batches=min_batches,
            min_seconds=min_seconds,
            warmup_batches=warmup,
        )
