import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class EEG_Encoder(nn.Module):
    # change the input n_channels before using
    def __init__(self, batch_size=128, seq_len=384, n_channels=32, n_classes=2):
        super(EEG_Encoder, self).__init__()
        F1, D = 8, 2
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Layer 1 - Conv2D + BatchNorm
        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=F1,
                kernel_size=(1, 128),
                padding=(0, 64),
                bias=False),
            nn.BatchNorm2d(F1)
        )

        # Layer 2 - DepthwiseConv2D + AvgPool2D
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=F1,
                      out_channels=F1,
                      kernel_size=(17, 1),
                      groups=2,
                      bias=False),
            nn.ELU(),
            nn.Conv2d(in_channels=F1,
                      out_channels=F1 * D,
                      kernel_size=(16, 1),
                      groups=2,
                      bias=False),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(p=0.5)
        )

        # Layer 3 - SeparableConv2D + AvgPool2D
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=F1 * D,
                      out_channels=F1 * D,
                      kernel_size=(1, 48),
                      padding=(0, 8),
                      groups=16,
                      bias=False),
            nn.Conv2d(in_channels=F1 * D,
                      out_channels=F1 * D,
                      kernel_size=(1, 1),  # Pointwise
                      bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(p=0.5)
        )

    def get_feature(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        return x

    def forward(self, x):
        x = x.reshape(-1, 1, self.n_channels, self.seq_len)
        x = self.get_feature(x)
        x = x.view(x.size(0), -1)
        return x

# Encoder for EOG EMG
class EOG_Encoder(nn.Module):
    def __init__(self, batch_size=128, seq_len=384, n_channels=2, n_classes=2):
        super(EOG_Encoder, self).__init__()
        F1, D = 8, 2
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Layer 1 - Conv2D + BatchNorm
        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=F1,
                kernel_size=(1, 128),
                padding=(0, 64),
                bias=False),
            nn.BatchNorm2d(F1)
        )

        # Layer 2 - DepthwiseConv2D + AvgPool2D
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=F1,
                      out_channels=F1 * D,
                      kernel_size=(2, 1),
                      groups=2,
                      bias=False),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(p=0.5)
        )

        # Layer 3 - SeparableConv2D + AvgPool2D
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=F1 * D,
                      out_channels=F1 * D,
                      kernel_size=(1, 48),
                      padding=(0, 8),
                      groups=16,
                      bias=False),
            nn.Conv2d(in_channels=F1 * D,
                      out_channels=F1 * D,
                      kernel_size=(1, 1),  # Pointwise
                      bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(p=0.5)
        )

    def get_feature(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        return x

    def forward(self, x):
        x = x.reshape(-1, 1, self.n_channels, self.seq_len)
        x = self.get_feature(x)
        x = x.view(x.size(0), -1)
        return x

