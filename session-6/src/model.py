import torch
import torch.nn as nn


class RegressionModel(nn.Module):
    # You should build your model with at least 2 layers using tanh activation in between
    def __init__(self, input_size, hidden_size):
        super(RegressionModel, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=hidden_size),
            nn.Tanh(),
            nn.Linear(in_features=hidden_size, out_features=1)
        )

    def forward(self, x):
        y = self.mlp(x)
        return y
        