import torch.nn as nn
import torch
from torch.nn.modules.activation import LogSoftmax

class ConvBlock(nn.Module):

    def __init__(
            self,
            num_inp_channels: int,
            num_out_fmaps: int,
            kernel_size: int,
            dropout: float,
            pool_size: int=2) -> None:

        super().__init__()

        self.conv = nn.Conv2d(in_channels=num_inp_channels, out_channels=num_out_fmaps, kernel_size=(kernel_size,kernel_size))
        self.dropout = nn.Dropout2d(p=dropout)
        self.relu = nn.LeakyReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=(2,2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool(self.relu(self.conv(x)))

class MyModel(nn.Module):

    def __init__(self, layers_size: float=8, dropout: float=0.5) -> None:
        super().__init__()

        self.layers_size = layers_size
        self.dropout = dropout
        self.conv1 = ConvBlock(num_inp_channels=1, num_out_fmaps=layers_size, kernel_size=3, dropout=dropout)
        self.conv2 = ConvBlock(num_inp_channels=layers_size, num_out_fmaps=layers_size*2, kernel_size=3, dropout=dropout)
        self.conv3 = ConvBlock(num_inp_channels=layers_size*2, num_out_fmaps=layers_size*4, kernel_size=3, dropout=dropout)
        self.conv4 = ConvBlock(num_inp_channels=layers_size*4, num_out_fmaps=layers_size*8, kernel_size=3, dropout=dropout)

        self.mlp = nn.Sequential(
            nn.Linear(in_features=32*layers_size, out_features=16*layers_size),
            nn.Dropout(p=dropout),
            nn.LeakyReLU(),
            nn.Linear(in_features=16*layers_size, out_features=8*layers_size),
            nn.Dropout(p=dropout),
            nn.LeakyReLU(),
            nn.BatchNorm1d(num_features=layers_size*8),
            nn.Linear(in_features=8*layers_size, out_features=15),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        bsz, nch, height, width = x.shape

        x = torch.reshape(x, (bsz, (nch * height * width)))

        y = self.mlp(x)

        return y





