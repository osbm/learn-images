import torch
from torch import nn

class GaussianFourierFeatures(nn.Module):
    def __init__(self, mapping_size=256, scale=10, input_vector_size=2):
        super().__init__()
        B = torch.normal(0, 1, size=(mapping_size, input_vector_size)) * scale
        self.B = nn.Parameter(B, requires_grad=False)
        self.output_dim = mapping_size * 2

    def __call__(self, x):
        x_proj = (2.*torch.pi*x) @ self.B.T
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class FourierFeatures(nn.Module):
    def __init__(self, fourier_order=4, exponential_increase=False, multiply_by_2pi=False):
        super().__init__()
        self.order = fourier_order
        self.output_dim = fourier_order * 2 # for one sine and one cosine

        orders = torch.arange(1, self.order + 1).float()
        if exponential_increase:
            orders *= orders

        self.orders = nn.Parameter(orders, requires_grad=False) # make it a parameter so it can be saved and loaded to GPU by model.to(device)
        self.multiplying_factor = 2 * torch.pi if multiply_by_2pi else 1

    def forward(self,x):
        x = x.unsqueeze(-1)  # add an extra dimension for broadcasting
        x = self.multiplying_factor * self.orders * x
        features =  torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        features = features.flatten(-2, -1)
        return features

class Fourier2DFeatures(nn.Module):
    def __init__(self, fourier_order=4, exponential_increase=True, multiply_by_2pi=True):
        super().__init__()
        self.output_dim = (fourier_order ** 2) * 4

        orders = torch.arange(1, fourier_order + 1).float()
        if exponential_increase:
            orders *= orders

        self.multiplying_factor = 2 * torch.pi if multiply_by_2pi else 1
        self.orders = nn.Parameter(orders, requires_grad=False)

    def forward(self,x):
        features = []
        x_first_channel = self.multiplying_factor * x[:, :, 0]
        x_second_channel = self.multiplying_factor * x[:, :, 1]
        for n in self.orders:
            for m in self.orders:
                features.append((torch.cos(n*x_first_channel)*torch.cos(m*x_second_channel)).unsqueeze(-1))
                features.append((torch.cos(n*x_first_channel)*torch.sin(m*x_second_channel)).unsqueeze(-1))
                features.append((torch.sin(n*x_first_channel)*torch.cos(m*x_second_channel)).unsqueeze(-1))
                features.append((torch.sin(n*x_first_channel)*torch.sin(m*x_second_channel)).unsqueeze(-1))
        fourier_features = torch.cat(features, -1)
        return fourier_features
