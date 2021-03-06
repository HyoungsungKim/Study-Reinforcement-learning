import torch
import torch.nn as nn
import numpy as np


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, shape):
        return shape.view(shape.size()[0], -1)


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        conv_out_size = self._get_conv_out(input_shape)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, input_shape):
        # * To make 3D shape, we have to put 1 in torch.zeros
        # * torch.zeros(1, 1, 52, 52)
        # * -> troch.zeros(batch_dimension, *shape)
        out = self.conv(torch.zeros(1, *input_shape))
        return int(np.prod(out.size()))

    def forward(self, x):
        conv_out = self.conv(x)
        fc_out = self.fc(conv_out)
        return fc_out


if __name__ == "__main__":
    image = torch.rand(1, 52, 52)
    dqn = DQN(image.shape, 10)
