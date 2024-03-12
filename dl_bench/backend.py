import numpy as np
import torch
from torch.nn import Module


def str_to_dtype(dtype: str):
    if dtype == "float32":
        return torch.float32
    elif dtype == "bfloat16":
        return torch.bfloat16
    elif dtype == "int8":
        return torch.qint8
    else:
        raise ValueError(f"Unsupported data type: {dtype}")


def print_cpu_capability():
    try:
        text = str(torch.backends.cpu.get_cpu_capability())
    except:
        text = "N/A"
    print("Torch cpu capability:", text)


class Backend:
    def __init__(self, device, compiler, dtype="float32") -> None:
        print_cpu_capability()

        self.device_name = device
        self.device = self._get_device(device_name=device)
        self.compile_mode = compiler
        self.dtype = str_to_dtype(dtype)

    def to_device(self, x: torch.Tensor):
        if self.device_name in ("cuda", "xpu"):
            return x.to(self.device)
        elif self.device_name == "hpu":
            import habana_frameworks.torch.core as htcore

            return x.to(self.device)
        elif self.device_name == "cpu":
            return x
        else:
            raise ValueError("Unknown device")

    def sync(self):
        if self.device_name == "cuda":
            torch.cuda.synchronize()
        elif self.device_name == "hpu":
            import habana_frameworks.torch.core as htcore
            import habana_frameworks.torch as ht

            ht.hpu.synchronize()

            htcore.mark_step()

    def prepare_eval_transformer(self, model):
        # model = model.to(memory_format=torch.channels_last)

        model.eval()
        model.to(self.device)
        with torch.no_grad():
            model.eval()
            return self._compile_transformer_model(
                self.compile_mode, model, dtype=self.dtype
            )

    @staticmethod
    def _compile_transformer_model(compile_mode, model, dtype=torch.bfloat16):
        compile_mode = compile_mode.lower()
        # Empty string means no compilation
        if compile_mode == "torch":
            compiled_model = model
        elif compile_mode == "torchscript":
            raise NotImplementedError()
            compiled_model = torch.jit.trace(model, sample_input)
            compiled_model = torch.jit.freeze(compiled_model)
            print("Compiled with torchscript")
        elif compile_mode == "torchscript_onednn":
            raise NotImplementedError()
            # enable oneDNN graph fusion globally
            torch.jit.enable_onednn_fusion(True)
            compiled_model = torch.jit.trace(model, sample_input)
            compiled_model = torch.jit.freeze(compiled_model)
            print("Compiled with torchscript onednn")
        elif compile_mode == "ipex":
            import intel_extension_for_pytorch as ipex

            params = {} if dtype == torch.float32 else {"dtype": dtype}
            # compiled_model = ipex.llm.optimize(model, **params, inplace=True, deployment_mode=True)
            compiled_model = ipex.llm.optimize(model, **params)
            # compiled_model = ipex.optimize_transformers(model, **params)
            print("Compiled with ipex")
        elif compile_mode == "ipex_onednn_graph":
            raise NotImplementedError()
            print("Compiled with ipex_onednn_graph")
        elif compile_mode == "dynamo":
            compiled_model = torch.compile(
                model, fullgraph=True, dynamic=False, mode="reduce-overhead"
            )
            print("Compiled with dynamo")
        elif compile_mode == "torch_mlir_default_inference":
            raise NotImplementedError()
        elif compile_mode == "torch_mlir":
            raise NotImplementedError()
        else:
            raise ValueError(f"Unsupported mode {compile_mode}")

        return compiled_model

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
            compiled_model = torch.jit.freeze(compiled_model)
            print("Compiled with torchscript")
        elif compile_mode == "torchscript_onednn":
            # enable oneDNN graph fusion globally
            torch.jit.enable_onednn_fusion(True)
            compiled_model = torch.jit.trace(model, sample_input)
            compiled_model = torch.jit.freeze(compiled_model)
            print("Compiled with torchscript onednn")
        elif compile_mode == "ipex":
            import intel_extension_for_pytorch as ipex

            params = {} if dtype == torch.float32 else {"dtype": dtype}
            compiled_model = ipex.optimize(model, sample_input=sample_input, **params)
            print("Compiled with ipex")
        elif compile_mode == "ipex_onednn_graph":
            import intel_extension_for_pytorch as ipex
            from intel_extension_for_pytorch.quantization import prepare, convert

            # need to set_llga_fp32_bf16_enabled as False, when benchmark int8 dtype
            ipex._C.set_llga_fp32_bf16_enabled(True)
            model.eval()
            if dtype == torch.qint8:
                qconfig_mapping = ipex.quantization.default_static_qconfig_mapping
                prepared_model = prepare(
                    model, qconfig_mapping, example_inputs=sample_input, inplace=False
                )
                convert_model = convert(prepared_model)
                compiled_model = torch.jit.trace(convert_model, sample_input)
                compiled_model = torch.jit.freeze(compiled_model)
            elif dtype == torch.bfloat16:
                with torch.cpu.amp.autocast(enabled=True, dtype=dtype), torch.no_grad():
                    compiled_model = torch.jit.trace(model, sample_input)
                    compiled_model = torch.jit.freeze(compiled_model)
            else:
                compiled_model = torch.jit.trace(model, sample_input)
                compiled_model = torch.jit.freeze(compiled_model)
            print("Compiled with ipex_onednn_graph")
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
        elif compile_mode == "torch_mlir" or compile_mode == "torch_mlir_xsmm":
            from torch_mlir._dynamo_fx_importer import import_fx_graph_as_func
            from torch_mlir_e2e_test.configs.torchdynamo import jit
            from torch_mlir_e2e_test.framework import TestOptions

            # from torch_mlir_e2e_test.linalg_on_tensors_backends.refbackend import RefBackendLinalgOnTensorsBackend
            from torch_mlir_e2e_test.linalg_on_tensors_backends.cpuprotobackend import (
                CpuProtoLinalgOnTensorsBackend,
            )
            from torch_mlir_e2e_test.linalg_on_tensors_backends.xsmmprotobackend import (
                XsmmProtoLinalgOnTensorsBackend,
            )
            import torch.utils._pytree as pytree

            # debug_timer seems to cause problems:
            # TypeError: TestOptions.__init__() got an unexpected keyword argument 'debug_timer'
            # opts = TestOptions(debug_timer=False, use_kernels=True)
            opts = TestOptions(use_kernels=True)
            module = jit(
                model,
                [sample_input],
                "test_name",
                opts,
                output_type="linalg-on-tensors",
            )
            backend = (
                CpuProtoLinalgOnTensorsBackend(opts)
                if compile_mode == "torch_mlir"
                else XsmmProtoLinalgOnTensorsBackend(opts)
            )
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
        if device_name == "xpu":
            return "xpu"
        elif device_name in ("cpu", "xpu", "cuda"):
            return torch.device(device_name)
        elif device_name == "hpu":
            import habana_frameworks.torch.core as htcore

            return torch.device(device_name)
        elif device_name == "openvino-cpu" or device_name == "openvino-gpu":
            from torch_ort import ORTInferenceModule, OpenVINOProviderOptions

            raise NotImplementedError("Openvino not ready yet.")
        else:
            raise ValueError(f"Unknown execution device {device_name}.")


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


def refine_result_type(_result):
    if isinstance(_result, tuple):
        return tuple(refine_result_type(x) for x in _result)
    elif isinstance(_result, np.ndarray):
        return torch.from_numpy(_result)
    elif isinstance(_result, (bool, int, float)):
        return _result
    else:
        raise ValueError(f"Unhandled return type {type(_result)}")
