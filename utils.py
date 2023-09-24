from PIL import Image
import torch

def generate_lin_space(image_size=(28, 28)):
    # generate 2d linear space of coordinates
    # numbers will be between 0 and 1

    X = torch.linspace(0, 1, image_size[0])
    Y = torch.linspace(0, 1, image_size[1])
    grid = torch.stack(torch.meshgrid(X, Y, indexing='ij'))
    grid = grid.permute(1, 2, 0)
    # merge the two dimensions
    grid = grid.reshape(-1, 2)
    return grid

