import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .backend import Backend


def get_time():
    return time.perf_counter()


class RandomInfDataset(Dataset):
    def __init__(self, n, in_shape, seed=42):
        super().__init__()
        np.random.seed(seed)

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
        ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory
    )
    return train_loader, test_loader


def get_report(fw_times, duration_s, n_items, flops_per_sample):
    return {
        "duration_s": duration_s,
        "samples_per_s": n_items / sum(fw_times),
        "samples_per_s_dirty": n_items / duration_s,
        "flops_per_sample": flops_per_sample,
        "n_items": n_items,
        "p00": np.percentile(fw_times, 0),
        "p50": np.percentile(fw_times, 50),
        "p90": np.percentile(fw_times, 90),
        "p100": max(fw_times),
    }


class Benchmark:
    def __init__(
        self,
        net,
        in_shape,
        dataset,
        batch_size,
        min_batches=10,
        min_seconds=10,
        warmup_batches=3,
    ) -> None:
        self.model = net
        self.in_shape = in_shape
        self.dataset = dataset
        self.batch_size = batch_size
        self.warmup_batches = warmup_batches
        self.min_batches = min_batches
        self.min_seconds = min_seconds

    def compile(self, sample, backend: Backend):
        self.model = backend.prepare_eval_model(self.model, sample_input=sample)

    def inference(self, backend: Backend):
        # timout if running for more than 3 minutes already
        max_time = 180

        test_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=backend.device_name == "cuda",
        )

        try:
            print("Torch cpu capability:", torch.backends.cpu.get_cpu_capability())
        except:
            pass

        flops_per_sample = get_macs(self.model, self.in_shape, backend) * 2

        sample = next(iter(test_loader))
        self.compile(sample, backend)

        n_items = 0
        outputs = []
        fw_times = []

        self.model.eval()
        with torch.inference_mode():
            start = get_time()
            for i, x in enumerate(test_loader):
                backend.sync()
                s = get_time()
                x = backend.to_device(x)
                if backend.dtype != torch.float32:
                    with torch.autocast(
                        device_type=backend.device_name,
                        dtype=backend.dtype,
                    ):
                        y = self.model(x)
                else:
                    y = self.model(x)

                backend.sync()

                if i < self.warmup_batches:
                    # We restart timer because that was just a warmup
                    start = time.perf_counter()
                    continue

                fw_times.append(get_time() - s)
                n_items += len(x)
                outputs.append(y)

                # early stopping if we have 10+ batches and were running for 10+ seconds
                if (
                    (time.perf_counter() - start) > self.min_seconds
                    and n_items >= self.batch_size * self.min_batches
                ):
                    break

                if (get_time() - start) > max_time:
                    break

        stop = get_time()

        report = get_report(
            fw_times=fw_times,
            duration_s=stop - start,
            n_items=n_items,
            flops_per_sample=flops_per_sample,
        )
        return report, outputs

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


def get_macs(model, in_shape, backend):
    """Calculate MACs, conventional FLOPS = MACs * 2."""
    from ptflops import get_model_complexity_info

    model.eval()
    with torch.no_grad():
        macs, params = get_model_complexity_info(
            model, in_shape, as_strings=False, print_per_layer_stat=False, verbose=True
        )
    return macs
