import time

import torch
from torch.nn import Module, Linear
import torch.nn.functional as F

from dl_bench.utils import Benchmark
from dl_bench.bench.mlp import RandomInfDataset
from typing import List


class MLP(Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_l_dims: List):
        super().__init__()
        self.input_fc = Linear(input_dim, hidden_l_dims[0])
        self.hidden_fc = Linear(*hidden_l_dims)
        self.output_fc = Linear(hidden_l_dims[1], output_dim)

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
    def __init__(self, params) -> None:
        INPUT_H = 28
        INPUT_W = 28
        OUTPUT_DIM = 10
        HIDDEN_LAYER_DIMS = (250, 100)

        in_shape = (INPUT_H, INPUT_W)
        batch_size = 1

        dataset = RandomInfDataset(1, in_shape)

        net = MLP(INPUT_H * INPUT_W, OUTPUT_DIM, HIDDEN_LAYER_DIMS)
        super().__init__(
            net=net, in_shape=in_shape, dataset=dataset, batch_size=batch_size
        )


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
        validate_accuracy.validate_accuracy(
            model, dtype, criterion, validation_loader, device, lossv, accv
        )
