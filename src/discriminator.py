import torch.nn as nn
from torch.autograd import Function
import torch


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = -ctx.alpha * grad_output
        return output, None

class GradientReversalLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, alpha = 1.0):
        return GradientReversalFunction.apply(x, alpha)


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_domains=4):
        super().__init__()

        layers = [nn.Linear(input_dim, hidden_dim),
                  nn.ReLU(),
                  nn.Linear(hidden_dim, num_domains)]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)