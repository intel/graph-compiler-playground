from dl_bench.benchmark import Benchmark, RandomInfDataset


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
        "mobilenet_v3l": mobilenet_v3_large,
    }
    if name in name2model:
        return name2model[name]()
    else:
        raise ValueError(f"Unknown name {name}")


class CnnBenchmark(Benchmark):
    def __init__(self, params) -> None:
        batch_size = int(params.get("batch_size", 1024))

        in_shape = (3, 224, 224)
        min_batches = 10
        min_seconds = 20
        warmup = 10
        DATASET_SIZE = max(10_240, batch_size * (min_batches + warmup))
        dataset = RandomInfDataset(DATASET_SIZE, in_shape)

        name = params.get("name", "resnet50")
        net = get_cnn(name=name)

        super().__init__(
            net=net,
            in_shape=in_shape,
            dataset=dataset,
            batch_size=batch_size,
            min_batches=min_batches,
            warmup_batches=warmup,
            min_seconds=min_seconds,
        )
