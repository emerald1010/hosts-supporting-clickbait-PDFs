import torch
from torch import nn, flatten

from Models.Classification.BaseClassification import BaseClassificationModel


class DownsamplingCNN(BaseClassificationModel):

    def __init__(self):
        super(DownsamplingCNN, self).__init__(1)

        self.feature_extraction = nn.Sequential(
            DownBlock(2, 32, True),
            DownBlock(32, 64, True),
            DownBlock(64, 64, True),
            DownBlock(64, 32, True),
            DownBlock(32, 16, True),
            DownBlock(16, 8, True)
        )
        self.fc_1 = nn.Linear(512, 32)
        self.act = nn.ReLU()
        self.fc_2 = nn.Linear(32, 1)

    def forward(self, x, **kwargs):
        x = self.feature_extraction(x)
        x = flatten(x, start_dim=1)
        x = self.fc_1(x)
        x = self.act(x)
        x = self.fc_2(x)
        return torch.squeeze(x, dim=1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)


class DownBlock(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, use_dropout=True):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, use_dropout)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class ConvBlock(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, use_dropout=True):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        if use_dropout:
            self.conv_block.add_module("dropout", nn.Dropout(0.2))

    def forward(self, x):
        return self.conv_block(x)
