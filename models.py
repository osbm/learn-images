import torch
from torch import nn

class SimpleMLP(nn.Module):
    """ 
    Very simple linear torch model. Uses relu activation

    Parameters: 
    hidden_size (float): number of parameters per hidden layer
    num_hidden_layers (float): number of hidden layers
    init_size (float): number of parameters in input layer
    output_size (float): number of parameters in output layer
    """
    def __init__(self, hidden_size=100, num_hidden_layers=7, init_size=2, output_size=1):
        super().__init__()
        layers = [
            nn.Linear(init_size, hidden_size),
            nn.ReLU()
        ]
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, output_size))
        self.seq = nn.Sequential(*layers)

    def forward(self,x):
        x = self.seq(x)
        return torch.sigmoid(x)
        # return x
