import torch
from torch.nn import Module, Linear
import torch.nn.functional as F

from utils import Benchmark, select_execution_engine, train

HIDDEN_LAYER_DIMS = (250, 100)
INPUT_DIM = 28 * 28
OUTPUT_DIM = 10


class MLP(Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_fc = Linear(input_dim, HIDDEN_LAYER_DIMS[0])
        self.hidden_fc = Linear(*HIDDEN_LAYER_DIMS)
        self.output_fc = Linear(HIDDEN_LAYER_DIMS[1], output_dim)

    def forward(self, x):
        # x = [batch size, height, width]
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        h_1 = F.relu(self.input_fc(x))
        # h_1 = [batch size, HIDDEN_LAYER_DIMS[0]]
        h_2 = F.relu(self.hidden_fc(h_1))
        # h_2 = [batch size, HIDDEN_LAYER_DIMS[1]]
        y_pred = self.output_fc(h_2)
        # y_pred = [batch size, output dim]
        return y_pred, h_2


class MPLBenchmark(Benchmark):
    def __init__(self, engine: str, jit: bool):
        model = MLP(INPUT_DIM, OUTPUT_DIM)
        self.model, self.device = select_execution_engine(engine, model)
        self._jit = jit

    def execute(self, need_train):
        # model.load_state_dict(torch.load('tut1-model.pt'))

        # enable oneDNN graph fusion globally
        torch.jit.enable_onednn_fusion(True)
        import os

        os.environ["ONEDNN_GRAPH_DUMP"] = "graph"
        # os.environ["DNNL_VERBOSE"]="1"
        # os.environ["PYTORCH_JIT_LOG_LEVEL"]=">>graph_helper:>>graph_fuser:>>kernel:>>interface"

        rand_inp = torch.rand(1, 1, 28, 28).to(self.device)
        # construct the model
        with torch.no_grad():
            self.model.eval()
            if self._jit == "TorchScript":
                compiled_model = torch.jit.trace(self.model, rand_inp)
            elif self._jit == "IPEX":
                import intel_extension_for_pytorch

                compiled_model = intel_extension_for_pytorch.optimize(self.model)
            elif self._jit == "Dynamo":
                compiled_model = torch.compile(self.model)
            elif self._jit == "TorchMLIR":
                import torch._dynamo as dynamo
                from dynamo_be import torch_mlir_be as be

                self.model.train(False)
                compiled_model = dynamo.optimize(be.refbackend_torchdynamo_backend)(
                    self.model
                )
            else:
                compiled_model = self.model

        # print(compiled_model)
        compiled_model(rand_inp)

        # run the model
        with torch.no_grad():
            # oneDNN graph fusion will be triggered during runtime
            output = compiled_model(rand_inp)
        if False:
            # Validate on mnist more comlex, there is an error in torch-mlir dyanmo
            import tools.validate_accuracy as tva

            loss_vector = []
            accuracy_vector = []
            criterion = torch.nn.CrossEntropyLoss()
            tva.validate_accuracy_mnist(
                compiled_model,
                torch.float,
                criterion,
                self.device,
                loss_vector,
                accuracy_vector,
            )
        if need_train:
            train(compiled_model, self.device)

        print(output)
