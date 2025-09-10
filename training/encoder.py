import torch.nn as nn

class Encoder(nn.Module):
    def __init__(
        self, 
        input_dim,
        hidden_dims=[128, 64], 
        kernel_sizes=[2, 2]
    ):
        super(Encoder, self).__init__()

        self.conv1d_1 = nn.Conv1d(
            in_channels=input_dim, 
            out_channels=hidden_dims[0], 
            kernel_size=kernel_sizes[0]
        )

        self.conv1d_2 = nn.Conv1d(
            in_channels=hidden_dims[0], 
            out_channels=hidden_dims[1], 
            kernel_size=kernel_sizes[1]
        )

        self.pooling = nn.MaxPool1d(kernel_size=kernel_sizes[1])
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv1d_1(x)
        x = self.conv1d_2(x)
        x = self.pooling(x)
        x = self.flatten(x)

        return x
