import torch.nn as nn

class Projector(nn.Module):
    """Projector that maps input_dim to the shared latent space's dimension"""
    def __init__(self, input_dim, shared_dim, normalize = True):
        super().__init__()
        self.linear = nn.Linear(input_dim, shared_dim)
        self.normalize = normalize

    
    def forward(self, x):
        x = self.linear(x)
        if self.normalize:
            x = nn.functional.normalize(x, p = 2, dim = 1)
        return x