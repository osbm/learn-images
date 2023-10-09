import torch
from PIL import Image
import numpy as np

def generate_lin_space(image_size=(28, 28), flatten_xy=False):
    # generate 2d linear space of coordinates
    # numbers will be between 0 and 1

    X = torch.linspace(0, 1, image_size[0])
    Y = torch.linspace(0, 1, image_size[1])
    grid = torch.stack(torch.meshgrid(X, Y, indexing='ij'))
    grid = grid.permute(1, 2, 0)

    if flatten_xy:
        grid = grid.flatten(0, 1)

    return grid

def get_target_tensor(file_path: str="data/target.jpeg", map_between_minus_one_and_one: bool=False, flatten_xy: bool=False, also_return_image_size: bool=False):
    target_tensor = Image.open(file_path)
    target_tensor = torch.from_numpy(np.array(target_tensor))
    image_size = target_tensor.shape
    target_tensor = target_tensor.float()
    target_tensor = target_tensor / 255

    if map_between_minus_one_and_one:
        target_tensor = target_tensor * 2 - 1

    if flatten_xy:
        target_tensor = target_tensor.flatten(0, 1)

    if also_return_image_size:
        return target_tensor, image_size
    else:
        return target_tensor

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
