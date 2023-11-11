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

        self.conv = nn.Conv2d(
            in_channels=num_inp_channels,
            out_channels=num_out_fmaps,
            kernel_size=kernel_size,
        )
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=pool_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        convolution = self.conv(x)
        relu = self.relu(convolution)
        return self.maxpool(relu)


class MyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # self.pad = nn.ConstantPad2d(2, 0)

        self.conv1 = ConvBlock(num_inp_channels=1, num_out_fmaps=6, kernel_size=5)
        self.conv2 = ConvBlock(num_inp_channels=6, num_out_fmaps=16, kernel_size=5)
        self.conv3 = ConvBlock(num_inp_channels=16, num_out_fmaps=32, kernel_size=5)

        self.mlp = nn.Sequential(
            # nn.Linear(400, 120),
            # nn.ReLU(),
            nn.Linear(512, 84),  # Which is the number?
            nn.ReLU(),
            nn.Linear(84, 15),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = self.pad(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Obtain the parameters of the tensor in terms of:
        # 1) batch size
        # 2) number of channels
        # 3) spatial "height"
        # 4) spatial "width"
        bsz, nch, height, width = x.shape
        # TODO: Flatten the feature map with the reshape() operator
        # within each batch sample
        x = torch.reshape(
            x, (bsz, nch * height * width)
        )  # Es pot ficar -1 en lloc de ficar (bsz, nch * height * width) per aplanar a un vector de 1D

        # print(x.shape)

        y = self.mlp(x)
        return y
