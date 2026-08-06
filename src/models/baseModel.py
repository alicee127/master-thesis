import torch
import torch.nn as nn

class BaseModel(nn.Module):

    def __init__(self, target_sr, output_dim):
        super().__init__()
        self.target_sr = target_sr
        self.output_dim = output_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def freeze(self, module):
        for param in module.parameters():
            param.requires_grad = False
        module.eval()

    def encode(self):
        #define in each subclass
        raise NotImplementedError
    
