import torch.nn as nn
import torch

class Classifier(nn.Module):

    def __init__(self, shared_dim, hidden_dim = 64, dropout = 0.0):
        super().__init__()

        layers = [nn.Linear(shared_dim, hidden_dim),
                  nn.ReLU()]

        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, 1)) #the output dimension is 1 because it performs binary classification

        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

    def predict_proba(self, x):
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))

    def predict(self, x):
        return (self.predict_proba(x) >0.5).long()
