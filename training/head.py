import torch.nn as nn

class ReconstructionHead(nn.Module):
    def __init__(self, encoder, out_dim):
        super(ReconstructionHead, self).__init__()
        
        self.encoder = encoder
        self.linear = nn.Linear(
            encoder.flatten.out_features, out_dim
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.linear(x)

        return x
