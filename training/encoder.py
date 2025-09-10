import torch.nn as nn
import torch

class Encoder(nn.Module):
    def __init__(
        self, 
        input_dim,
        hidden_dims=[128, 64], 
        kernel_sizes=[2, 2],
    ):
        super(Encoder, self).__init__()

        self.conv1d_1 = nn.Conv1d(
            in_channels=input_dim, 
            out_channels=hidden_dims[0], 
            kernel_size=kernel_sizes[0]
        )
        self.act1 = nn.ReLU()

        self.conv1d_2 = nn.Conv1d(
            in_channels=hidden_dims[0], 
            out_channels=hidden_dims[1], 
            kernel_size=kernel_sizes[1]
        )
        self.act2 = nn.ReLU()

        self.pooling = nn.MaxPool1d(kernel_size=kernel_sizes[1])
        self.flatten = nn.Flatten()

        with torch.no_grad():
            dummy = torch.zeros(1, 10, input_dim)
            out = self.forward(dummy)
            self.flattened_dim = out.view(1, -1).shape[1]

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1d_1(x)
        x = self.act1(x)
        x = self.conv1d_2(x)
        x = self.act2(x)
        x = self.pooling(x)
        x = self.flatten(x)
        return x
