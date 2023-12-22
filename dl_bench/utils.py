import abc
import copy
import time
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset


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
        ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=pin_memory
    )
    return train_loader, test_loader


def recursively_convert_to_numpy(o: Any):
    if isinstance(o, torch.Tensor):
        return o.numpy()
    if isinstance(o, tuple):
        return tuple(recursively_convert_to_numpy(x) for x in o)
    if isinstance(o, list):
        return [recursively_convert_to_numpy(x) for x in o]
    if isinstance(o, dict):
        return {k: recursively_convert_to_numpy(v) for k, v in o.items()}
    # No-op cases. Explicitly enumerated to avoid things sneaking through.
    if isinstance(o, str):
        return o
    if isinstance(o, float):
        return o
    if isinstance(o, int):
        return o
    raise Exception(f"Unexpected Python function input: {o}")


def recursively_convert_from_numpy(o: Any):
    if isinstance(o, np.ndarray):
        return torch.from_numpy(o)
    if isinstance(o, tuple):
        return tuple(recursively_convert_from_numpy(x) for x in o)
    if isinstance(o, list):
        return [recursively_convert_from_numpy(x) for x in o]
    if isinstance(o, dict):
        return {k: recursively_convert_from_numpy(v) for k, v in o.items()}
    # No-op cases. Explicitly enumerated to avoid things sneaking through.
    if isinstance(o, str):
        return o
    if isinstance(o, float):
        return o
    if isinstance(o, int):
        return o
    raise Exception(f"Unexpected Python function output: {o}")


def refine_result_type(_result):
    if isinstance(_result, tuple):
        return tuple(refine_result_type(x) for x in _result)
    elif isinstance(_result, np.ndarray):
        return torch.from_numpy(_result)
    elif isinstance(_result, (bool, int, float)):
        return _result
    else:
        raise ValueError(f"Unhandled return type {type(_result)}")


class TimerManager:
    def __init__(self):
        self.name2time = {}
        self.name = None

    def timeit(self, name):
        self.name = name
        return self

    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        assert self.start is not None and self.name is not None
        self.time = perf_counter() - self.start
        self.readout = f"Time: {self.time:.3f} seconds"
        self.name2time[self.name] = perf_counter() - self.start
        self.start = None
        self.name = None

    def get_results(self):
        return self.name2time


def str_to_dtype(dtype: str):
    if dtype == "float32":
        return torch.float32
    elif dtype == "bfloat16":
        return torch.bfloat16
    else:
        raise ValueError(f"Unsupported data type: {dtype}")


class Backend:
    def __init__(self, device, compiler, dtype="float32") -> None:
        self.device_name = device
        self.device = self._get_device(device_name=device)
        self.compile_mode = compiler
        self.dtype = str_to_dtype(dtype)

    def to_device(self, x: torch.Tensor):
        if self.device_name == "cuda":
            return x.to(self.device)
        elif self.device_name == "xpu":
            raise NotImplementedError("xpu have no to_device impl yet.")
        elif self.device_name == "cpu":
            return x
        else:
            raise ValueError("Unknown device")

    def prepare_eval_model(self, model, sample_input):
        model.to(self.device)
        with torch.no_grad():
            model.eval()
            return self._compile_model(
                self.compile_mode, self.device, model, sample_input, dtype=self.dtype
            )

    @staticmethod
    def _compile_model(compile_mode: str, device, model: Module, sample_input, dtype):
        sample_input = sample_input.to(device)

        # with torch.autocast(device_type=backend.device_name, dtype=backend.dtype):
        compile_mode = compile_mode.lower()
        # Empty string means no compilation
        if compile_mode == "torch":
            compiled_model = model
        elif compile_mode == "torchscript":
            compiled_model = torch.jit.trace(model, sample_input)
            print("Compiled with torchscript")
        elif compile_mode == "torchscript_onednn":
            # enable oneDNN graph fusion globally
            torch.jit.enable_onednn_fusion(True)
            compiled_model = torch.jit.trace(model, sample_input)
            print("Compiled with torchscript onednn")
        elif compile_mode == "ipex":
            import intel_extension_for_pytorch

            compiled_model = intel_extension_for_pytorch.optimize(model)
            print("Compiled with ipex")
        elif compile_mode == "dynamo":
            compiled_model = torch.compile(
                model, fullgraph=True, dynamic=False, mode="reduce-overhead"
            )
            print("Compiled with dynamo")
        elif compile_mode == "torch_mlir_default_inference":
            import torch._dynamo as dynamo
            from dl_bench.dynamo_be import torch_mlir_be as be

            compiled_model = dynamo.optimize(be.refbackend_torchdynamo_backend)(model)
            print("Compiled with torch_mlir (torchscript, inference)")
        elif compile_mode == "torch_mlir":
            from torch_mlir._dynamo_fx_importer import import_fx_graph_as_func
            from torch_mlir_e2e_test.configs.torchdynamo import jit
            from torch_mlir_e2e_test.framework import TestOptions

            # from torch_mlir_e2e_test.linalg_on_tensors_backends.refbackend import RefBackendLinalgOnTensorsBackend
            from torch_mlir_e2e_test.linalg_on_tensors_backends.cpuprotobackend import (
                CpuProtoLinalgOnTensorsBackend,
            )
            import torch.utils._pytree as pytree

            #            debug_timer seems to cause problems:
            #            TypeError: TestOptions.__init__() got an unexpected keyword argument 'debug_timer'
            #            opts = TestOptions(debug_timer=False, use_kernels=True)
            opts = TestOptions(use_kernels=True)
            module = jit(
                model,
                [sample_input],
                "test_name",
                opts,
                output_type="linalg-on-tensors",
            )
            backend = CpuProtoLinalgOnTensorsBackend(opts)
            # backend = RefBackendLinalgOnTensorsBackend()
            module = backend.compile(module)
            backend_module = backend.load(module)

            params = {
                **dict(model.named_parameters(remove_duplicate=False)),
                **dict(model.named_buffers(remove_duplicate=False)),
            }
            params_flat, params_spec = pytree.tree_flatten(params)
            params_flat = list(params_flat)

            class result:
                def __call__(self, *args):
                    numpy_inputs = recursively_convert_to_numpy(params_flat + [*args])
                    return refine_result_type(
                        getattr(backend_module, model.__class__.__name__)(*numpy_inputs)
                    )

                def eval(self):
                    pass

            compiled_model = result()
            print("Compiled with torch_mlir")
        elif compile_mode == "torch_mlir_xsmm":
            from torch_mlir._dynamo_fx_importer import import_fx_graph_as_func
            from torch_mlir_e2e_test.configs.torchdynamo import jit
            from torch_mlir_e2e_test.framework import TestOptions

            # from torch_mlir_e2e_test.linalg_on_tensors_backends.refbackend import RefBackendLinalgOnTensorsBackend
            from torch_mlir_e2e_test.linalg_on_tensors_backends.xsmmprotobackend import (
                XsmmProtoLinalgOnTensorsBackend,
            )
            import torch.utils._pytree as pytree

            #            debug_timer seems to cause problems:
            #            TypeError: TestOptions.__init__() got an unexpected keyword argument 'debug_timer'
            #            opts = TestOptions(debug_timer=False, use_kernels=True)
            opts = TestOptions()
            module = jit(
                model,
                [sample_input],
                "test_name",
                opts,
                output_type="linalg-on-tensors",
            )
            backend = XsmmProtoLinalgOnTensorsBackend(opts)
            # backend = RefBackendLinalgOnTensorsBackend()
            module = backend.compile(module)
            backend_module = backend.load(module)

            params = {
                **dict(model.named_parameters(remove_duplicate=False)),
                **dict(model.named_buffers(remove_duplicate=False)),
            }
            params_flat, params_spec = pytree.tree_flatten(params)
            params_flat = list(params_flat)

            class result:
                def __call__(self, *args):
                    numpy_inputs = recursively_convert_to_numpy(params_flat + [*args])
                    return refine_result_type(
                        getattr(backend_module, model.__class__.__name__)(*numpy_inputs)
                    )

                def eval(self):
                    pass

            compiled_model = result()
            print("Compiled with XSMM torch_mlir")
        else:
            raise ValueError(f"Unsupported mode {compile_mode}")

        return compiled_model

    @staticmethod
    def _get_device(device_name):
        device_name = device_name.lower()
        # TODO: do we really need this import?
        if device_name == "xpu":
            import intel_extension_for_pytorch

        if device_name in ("cpu", "xpu", "cuda"):
            device = torch.device(device_name)
            return device
        elif device_name == "openvino-cpu" or device_name == "openvino-gpu":
            from torch_ort import ORTInferenceModule, OpenVINOProviderOptions

            raise NotImplementedError("Openvino not ready yet.")
        else:
            raise ValueError(f"Unknown execution device {device_name}.")


class Benchmark:
    def __init__(
        self, net, in_shape, dataset, batch_size, min_batches=10, min_seconds=10
    ) -> None:
        self.net = net
        self.in_shape = in_shape
        self.dataset = dataset
        self.batch_size = batch_size
        self.min_batches = min_batches
        self.min_seconds = min_seconds

    def compile(self, sample, backend: Backend):
        self.net = backend.prepare_eval_model(self.net, sample_input=sample)

    def inference(self, backend: Backend):
        test_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=backend.device_name == "cuda",
        )

        tm = TimerManager()

        try:
            print("Torch cpu capability:", torch.backends.cpu.get_cpu_capability())
        except:
            pass

        self.flops_per_sample = get_macs(self.net, self.in_shape, backend) * 2

        sample = next(iter(test_loader))
        self.compile(sample, backend)

        print("Warmup started")
        with torch.no_grad():
            self.net.eval()
            with tm.timeit("warmup_s"):
                sample = backend.to_device(sample)
                self.net(sample)
                self.net(sample)
                self.net(sample)
        print("Warmup done")

        n_items = 0

        self.net.eval()
        outputs = []
        fw_times = []
        with torch.no_grad():
            start = time.perf_counter()
            # Duration is inconsistent now
            with tm.timeit("duration_s"):
                for i, x in enumerate(test_loader):
                    s = get_time()
                    x = backend.to_device(x)
                    if backend.dtype != torch.float32:
                        with torch.autocast(
                            device_type=backend.device_name,
                            dtype=backend.dtype,
                        ):
                            y = self.net(x)
                    else:
                        y = self.net(x)

                    fw_times.append(get_time() - s)
                    n_items += len(x)
                    outputs.append(y)

                    # early stopping
                    if (
                        (time.perf_counter() - start) > self.min_seconds
                        and n_items > self.batch_size * self.min_batches
                    ):
                        break

        print(
            f"Latency 0%-5%-50%-95%-100% are: {np.percentile(fw_times, [0, 5, 50, 95, 100])}"
        )

        results = tm.get_results()
        results["samples_per_s"] = n_items / sum(fw_times)
        results["flops_per_sample"] = self.flops_per_sample

        return results, outputs

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
