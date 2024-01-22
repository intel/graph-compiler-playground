from dl_bench.utils import Benchmark, RandomInfDataset

import torch

class Conv2dNoPaddingModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.conv = torch.nn.Conv2d(2, 10, 3, bias=False)
        self.train(False)

    def forward(self, x):
        return self.conv(x)

def get_op(name):

    name2model = {
        "conv210": Conv2dNoPaddingModule,
    }
    if name in name2model:
        return name2model[name]()
    else:
        raise ValueError(f"Unknown name {name}")


class OpsBenchmark(Benchmark):
    def __init__(self, params) -> None:
        batch_size = int(params.get("batch_size", 1024))

        in_shape = (2, 10, 20)
        min_batches = 10
        DATASET_SIZE = max(10_240, batch_size * min_batches)
        dataset = RandomInfDataset(DATASET_SIZE, in_shape)

        name = params.get("name", "conv210")
        net = get_op(name=name)

        super().__init__(
            net=net, in_shape=in_shape, dataset=dataset, batch_size=batch_size
        )
