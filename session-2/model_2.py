import torch.nn as nn
import torch


class ConvBlock(nn.Module):
    def __init__(
        self,
        num_inp_channels: int,
        num_out_fmaps: int,
        kernel_size: int,
        pool_size: int = 2,
    ) -> None:
        super().__init__()

        # TODO: define the 3 modules needed
        self.conv = nn.Conv2d(
            in_channels=num_inp_channels,
            out_channels=num_out_fmaps,
            kernel_size=kernel_size,
        )
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=pool_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(x.shape)
        return self.maxpool(self.relu(self.conv(x)))

class PseudoLeNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pad = nn.ConstantPad2d(2, 0)

        self.conv1 = ConvBlock(num_inp_channels=1, num_out_fmaps=6, kernel_size=5)
        self.conv2 = ConvBlock(num_inp_channels=6, num_out_fmaps=16, kernel_size=5)

        self.mlp = nn.Sequential(
            nn.Linear(3136, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 15),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pad(x)
        x = self.conv1(x)
        x = self.conv2(x)
        batch_size, num_ch, height, width = x.shape
        # Flatten the feature map with the reshape() operator
        # within each batch sample
        x = torch.reshape(x, (batch_size, num_ch * height * width))
        # This is the same as the previous line
        # x = torch.reshape(x, -1)

        y = self.mlp(x)
        return y
