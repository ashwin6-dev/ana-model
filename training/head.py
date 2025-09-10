import torch.nn as nn

class ReconstructionHead(nn.Module):
    def __init__(self, encoder, out_dim):
        super(ReconstructionHead, self).__init__()

        self.encoder = encoder
        self.linear1 = nn.Linear(
            encoder.flattened_dim, out_dim*2
        )
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(
            out_dim*2, out_dim
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        return x
