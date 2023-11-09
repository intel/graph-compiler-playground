import abc
import torch
from torch.nn import Module


class Benchmark(abc.ABC):
    @abc.abstractmethod
    def __init__(self, engine: str):
        pass

    @abc.abstractmethod
    def execute(self):
        pass


def select_execution_engine(engine: str, model: Module) -> Module:
    if engine == "CPU":
        device = torch.device("cpu")
        return model.to(device), device
    if engine == "IPEX-CPU" or engine == "IPEX-XPU":
        import intel_extension_for_pytorch

        device = torch.device("xpu" if engine == "IPEX-XPU" else "cpu")
        return model.to(device), device
    elif engine == "CUDA":
        device = torch.device("cuda")
        return model.to(device), device
    elif engine == "OPENVINO-CPU" or engine == "OPENVINO-GPU":
        from torch_ort import ORTInferenceModule, OpenVINOProviderOptions

        pass
    else:
        raise ValueError(f"Unknown execution engine {engine}.")
