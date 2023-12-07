import abc
from time import perf_counter

import torch
from torch.nn import Module


from typing import Any
import numpy as np


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


class Benchmark(abc.ABC):
    @abc.abstractmethod
    def run(self, backend, params):
        pass

def get_dtype(dtype):
    if dtype == 'float32':
        return torch.float32
    elif dtype == 'bfloat16':
        return torch.bfloat16
    else:
        raise ValueError(f"Unsupported data type: {dtype}")


class Backend:
    def __init__(self, device, compiler, dtype="float32") -> None:
        self.device_name = device
        self.device = self._get_device(device_name=device)
        self.compile_mode = compiler
        self.dtype = get_dtype(dtype)

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
