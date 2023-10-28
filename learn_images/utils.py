import torch
from PIL import Image
import numpy as np
import os
import matplotlib.image
from tqdm import tqdm
import shutil


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

def load_image(image_path, convert_to="L"):
    image = Image.open(image_path)
    image = image.convert(convert_to)
    image = np.array(image)
    image = torch.from_numpy(image)
    return image


def get_target_tensor(file_path: str="data/target.jpeg", convert_to="RGB", map_between_minus_one_and_one: bool=False, flatten_xy: bool=False, also_return_image_size: bool=False):
    target_tensor = load_image(image_path=file_path, convert_to=convert_to)
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


def save_model_outputs_as_frames(model, num_frames, device, output_folder):
    
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    
    # input linear space is between 0 and 1

    inputs = torch.linspace(0, 1, num_frames).view(-1, 1).to(device)

    with torch.no_grad():
        model.eval()
        outputs = model(inputs) # this is a big tensor of shape (num_frames, 360, 480)

    # reshape to (num_frames, 1, 360, 480)
    outputs = outputs.view(num_frames, 1, 360, 480)

    for i in tqdm(range(num_frames)):
        output = outputs[i].detach().cpu().numpy()
        output = output.reshape(360, 480)
        output = Image.fromarray(output)
        output.save(os.path.join(output_folder, f'{i}.png'))


def save_features_to_folder(tensor: torch.Tensor, output_folder: str="data/fourier_visualizations", file_prefix=None):
    '''
    tensor: torch.Tensor
    # tensor has to be 3 dimensional (height, width, channels)

    output_folder: str
    # output folder will be deleted if it already exists

    file_prefix: str or None
    # will be set to folder name if is None
    '''
    assert len(tensor.shape) == 3, "tensor has to be 3 dimensional (height, width, channel)"
    output_folder = str(output_folder)
    if file_prefix is None:
        file_prefix = output_folder.split(os.path.sep)[-1]
    
    if tensor.max() > 1 or tensor.min() < 0:
        print(f"Warning {file_prefix} has values outside 0, 1 range: min:{tensor.min()} max:{tensor.max()}")

    if os.path.exists(output_folder):
        print("Deleting the visualizations of previous runs.")
        shutil.rmtree(output_folder)
    
    os.makedirs(output_folder, exist_ok=True)
    
    for i in tqdm(range(tensor.shape[-1]), desc=f"Saving images for {file_prefix}"): # number of extracted features
        image = tensor[:, :, i].numpy()

        matplotlib.image.imsave(
            f"{output_folder}/{file_prefix}_{i}.png",
            image,
            cmap="gray",
        )

def get_model_size(model=None):
    total = 0
    for p in model.parameters():
        if p.requires_grad:
            total += p.numel() * p.element_size()

    return total
    
