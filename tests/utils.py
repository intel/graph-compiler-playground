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


def train(model: Module, device):
    from ..tools import load_mnist_dataset, train, validate_accuracy

    epochs = 15

    optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    train_loader, validation_loader = load_mnist_dataset()

    lossv, accv = [], []
    dtype = torch.float
    for epoch in range(1, epochs + 1):
        train(model, dtype, device, criterion, optimizer, epoch)
        validate_accuracy(
            model, dtype, criterion, validation_loader, device, lossv, accv
        )
