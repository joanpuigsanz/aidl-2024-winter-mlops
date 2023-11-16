import torch
import torch.nn as nn
import torch.nn.functional as F

# class ConvBlock(nn.Module):
#     def __init__(
#             self,
#             num_inp_channels: int,
#             num_out_fmaps: int,
#             kernel_size: int,
#             pool_size: int = 2,
#     ) -> None:
#         super().__init__()
#
#         self.conv = nn.Conv2d(in_channels=num_inp_channels, out_channels=num_out_fmaps, kernel_size=kernel_size)
#         self.relu = nn.ReLU()
#         self.maxpool = nn.MaxPool2d(kernel_size=pool_size)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         convolution = self.conv(x)
#         relu = self.relu(convolution)
#         return self.maxpool(relu)
#
#
# class MyModel(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.conv1 = ConvBlock(num_inp_channels=3, num_out_fmaps=6, kernel_size=5)
#         self.conv2 = ConvBlock(num_inp_channels=6, num_out_fmaps=16, kernel_size=5)
#         self.conv3 = ConvBlock(num_inp_channels=16, num_out_fmaps=32, kernel_size=5)
#
#         self.mlp = nn.Sequential(
#             # nn.Linear(400, 120), # If using 2 conv layers it needs  2704 inputs
#             # nn.ReLU(),
#             nn.Linear(512, 84),
#             nn.ReLU(),
#             nn.Linear(84, 15),
#             nn.LogSoftmax(dim=-1),
#         )
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#
#         batch_size, num_ch, height, width = x.shape
#         # Flatten the feature map with the reshape() operator
#         # within each batch sample
#         x = torch.reshape(x, (batch_size, num_ch * height * width))
#         # This is the same as the previous line
#         # x = torch.reshape(x, -1)
#
#         y = self.mlp(x)
#         return y

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(25088, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        batch_size, num_ch, height, width = x.shape
        x = torch.reshape(x, (batch_size, num_ch * height * width))
        # x = x.view(x.size(0), -1)
        x = self.fc_layers(x)

        return x
