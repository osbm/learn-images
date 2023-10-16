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
    def __init__(self, hidden_size=100, num_hidden_layers=7, init_size=2, output_size=1, output_activation="tanh"):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.output_size = output_size
        self.init_size = init_size
        self.output_activation = output_activation
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
        if self.output_activation == "sigmoid":
            return torch.sigmoid(x)
        elif self.output_activation == "tanh":
            return (torch.tanh(x)+1)/2

class SkipConnectionsMLP(nn.Module):
    def __init__(self, hidden_size=100, num_hidden_layers=7, init_size=2, output_size=3, output_activation="tanh"):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.output_size = output_size
        self.init_size = init_size
        self.output_activation = output_activation
        
        self.inLayer = nn.Linear(init_size, hidden_size)
        self.relu = nn.LeakyReLU()
        hidden = []
        for i in range(num_hidden_layers):
            in_size = hidden_size*2 + init_size if i>0 else hidden_size + init_size
            hidden.append(nn.Linear(in_size, hidden_size))
        self.hidden = nn.ModuleList(hidden)
        self.outLayer = nn.Linear(hidden_size*2+init_size, output_size)

    def forward(self, x):
        current = self.relu(self.inLayer(x))
        previous = torch.tensor([]).to(x.device)
        for layer in self.hidden:
            combined = torch.cat([current, previous, x], 1)
            previous = current
            current = self.relu(layer(combined))
        x = self.outLayer(torch.cat([current, previous, x], 1))
        if self.output_activation == "sigmoid":
            return torch.sigmoid(x)
        elif self.output_activation == "tanh":
            return (torch.tanh(x)+1)/2


class FrameGenerator(nn.Module):
    # this model will take in a vector and and output a 360x480 image 
    def __init__(self, input_size:int=1, hidden_layer_size: int= 128, output_size: tuple= (360, 480), num_hidden_layers=4, dropout=0.15):
        super(FrameGenerator, self).__init__()
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size
        self.num_hidden_layers=num_hidden_layers
        self.dropout = dropout

        self.layers = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_layer_size),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout),
            nn.BatchNorm1d(self.hidden_layer_size),
            *[nn.Sequential(
                nn.Linear(self.hidden_layer_size, self.hidden_layer_size),
                nn.LeakyReLU(),
                nn.Dropout(self.dropout),
                nn.BatchNorm1d(self.hidden_layer_size),
            ) for _ in range(self.num_hidden_layers)],
            nn.Linear(self.hidden_layer_size, self.output_size[0]*self.output_size[1]),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.view(-1, self.output_size[0], self.output_size[1])
        return x
