import torch
import torch.nn as nn

import numpy as np


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, shape):
        return shape.view(shape.size(0), -1)


class CNN(nn.Module):
    def __init__(self, shape, number_of_classes):
        super(CNN, self).__init__()

        # * Dropout will be deactivated while evaluation
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=shape[0], out_channels=16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.25),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.5),
            Flatten()
        )
        self.conv_out = self._get_conv_out(shape)

        self.fc = nn.Sequential(
            nn.Linear(self.conv_out, 64),
            nn.ReLU(),
            nn.Linear(64, number_of_classes),
            nn.BatchNorm1d(number_of_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, shape):
        conv = self.conv(shape)
        fc = self.fc(conv)
        return fc

    def _get_conv_out(self, shape):
        # * if shape is [3, 64, 64]
        # * torch.zeros(1, 3, 64, 64, 64)
        out = self.conv(torch.zeros(1, *shape))
        return int(np.prod(out.size()))
