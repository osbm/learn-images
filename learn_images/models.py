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

class FourierFeatues(nn.Module):
    def __init__(self, fourier_order=4):
        """ 
        Linear torch model that adds Fourier Features to the initial input x as \
        sin(x) + cos(x), sin(2x) + cos(2x), sin(3x) + cos(3x), ...
        

        Parameters: 
        fourier_order (int): number fourier features to use. Each addition adds 4x\
         parameters to each layer.
        hidden_size (float): number of non-skip parameters per hidden layer (SkipConn)
        num_hidden_layers (float): number of hidden layers (SkipConn)
        """
        super().__init__()
        self.fourier_order = fourier_order
        self.output_shape = fourier_order*4 + 2

    def forward(self,x):
        orders = torch.arange(1, self.fourier_order + 1).float().to(x.device)
        x = x.unsqueeze(-1)  # add an extra dimension for broadcasting
        fourier_features = torch.cat([torch.sin(orders * x), torch.cos(orders * x), x], dim=-1)
        fourier_features = fourier_features.view(x.shape[0], -1)  # flatten the last two dimensions
        return fourier_features

class PadeFeatures(nn.Module):
    def __init__(self, pade_order=4):
        """ 
        Linear torch model that adds Pade Features to the initial input x as \
        (x^2 + 1)/(x^2 + 2), (x^2 + 2)/(x^2 + 3), (x^2 + 3)/(x^2 + 4), ...
        

        Parameters: 
        pade_order (int): number pade features to use. Each addition adds 2x\
         parameters to each layer.
        hidden_size (float): number of non-skip parameters per hidden layer (SkipConn)
        num_hidden_layers (float): number of hidden layers (SkipConn)
        """
        super().__init__()
        self.pade_order = pade_order

    def forward(self,x):
        orders = torch.arange(1, self.pade_order + 1).float().to(x.device)
        x = x.unsqueeze(-1)  # add an extra dimension for broadcasting
        pade_features = torch.cat([(x**2 + orders), (x**2 + orders + 1)], dim=-1)
        pade_features = pade_features.view(x.shape[0], -1)  # flatten the last two dimensions
        return pade_features


class Fourier2DFeatures(nn.Module):
    def __init__(self, fourier_order=4):
        super().__init__()
        self.fourier_order = fourier_order
        self.output_shape = (fourier_order*fourier_order*4) + 2

    def forward(self,x):
        orders = torch.arange(0, self.fourier_order).float().to(x.device)
        features = [x]
        for n in orders:
            for m in orders:
                features.append((torch.cos(n*x[:,0])*torch.cos(m*x[:,1])).unsqueeze(-1))
                features.append((torch.cos(n*x[:,0])*torch.sin(m*x[:,1])).unsqueeze(-1))
                features.append((torch.sin(n*x[:,0])*torch.cos(m*x[:,1])).unsqueeze(-1))
                features.append((torch.sin(n*x[:,0])*torch.sin(m*x[:,1])).unsqueeze(-1))
        fourier_features = torch.cat(features, 1)
        return fourier_features
