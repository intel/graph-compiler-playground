import time

import torch
from torch.nn import Module, Linear
import torch.nn.functional as F

from utils import Backend, Benchmark

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


class MlpBasicBenchmark(Benchmark):
    def run(self, backend: Backend, params):
        need_train = params.get("need_train", False)
        rand_inp = torch.rand(1, 1, 28, 28).to(backend.device)

        model = MLP(INPUT_DIM, OUTPUT_DIM)

        compiled_model = backend.prepare_eval_model(model, rand_inp)

        with torch.no_grad():
            start_time = time.time()
            output = compiled_model(rand_inp)
            execution_time = round(time.time() - start_time, 2)
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
                backend.device,
                loss_vector,
                accuracy_vector,
            )
        if need_train:
            train(compiled_model, backend.device)

        print(output)

        if model != compiled_model:
            import tools.compare as cmp

            expected = model(rand_inp)
            cmp.compare(expected[0], output[0])

        return {"execution_time": execution_time}


def train(model: Module, device):
    from tools import train, validate_accuracy

    epochs = 2

    optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    train_loader, validation_loader = validate_accuracy.load_mnist_dataset()

    lossv, accv = [], []
    dtype = torch.float
    for epoch in range(1, epochs + 1):
        train.train(model, dtype, device, train_loader, criterion, optimizer, epoch)
        validate_accuracy.validate_accuracy(model, dtype, criterion, validation_loader, device, lossv, accv)
