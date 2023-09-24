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

class SkipConnectionsMLP(nn.Module):
    def __init__(self, hidden_size=100, num_hidden_layers=7, init_size=2, output_size=3):
        super().__init__()

        self.inLayer = nn.Linear(init_size, hidden_size)
        self.relu = nn.LeakyReLU()
        hidden = []
        for i in range(num_hidden_layers):
            in_size = hidden_size*2 + init_size if i>0 else hidden_size + init_size
            hidden.append(nn.Linear(in_size, hidden_size))
        self.hidden = nn.ModuleList(hidden)
        self.outLayer = nn.Linear(hidden_size*2+init_size, 3)

    def forward(self, x):
        current = self.relu(self.inLayer(x))
        previous = torch.tensor([]).to(x.device)
        for layer in self.hidden:
            combined = torch.cat([current, previous, x], 1)
            previous = current
            current = self.relu(layer(combined))
        y = self.outLayer(torch.cat([current, previous, x], 1))
        return (torch.tanh(y)+1)/2 # hey I think this works slightly better
        # return self.sig(y)
