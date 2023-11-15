import torch
from torch.nn import Module


class Backend:
    def __init__(self, device, compile_mode) -> None:
        self.device = self._get_device(device_name=device)
        self.compile_mode = compile_mode

    def prepare_eval_model(self, model, sample_input):
        model.to(self.device)
        with torch.no_grad():
            model.eval()
            return self._compile_model(self.compile_mode, model, sample_input)

    @staticmethod
    def _compile_model(compile_mode: str, model: Module, sample_input):
        compile_mode = compile_mode.lower()
        # Empty string means no compilation
        if compile_mode == "":
            compiled_model = model
        elif compile_mode == "torchscript":
            compiled_model = torch.jit.trace(model, sample_input)
        elif compile_mode == "torchscript_onednn":
            # enable oneDNN graph fusion globally
            torch.jit.enable_onednn_fusion(True)
            compiled_model = torch.jit.trace(model, sample_input)
        elif compile_mode == "ipex":
            import intel_extension_for_pytorch

            compiled_model = intel_extension_for_pytorch.optimize(model)
        elif compile_mode == "dynamo":
            compiled_model = torch.compile(model)
        elif compile_mode == "torch_mlir":
            import torch._dynamo as dynamo
            from dynamo_be import torch_mlir_be as be

            compiled_model = dynamo.optimize(be.refbackend_torchdynamo_backend)(model)
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
