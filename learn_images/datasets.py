import torch
from .utils import get_target_tensor, generate_lin_space, load_image
import os
import numpy as np


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_path="data/target.jpeg", convert_to="RBG", map_between_minus_one_and_one=True, flatten_xy=True):
        self.image_path = image_path
        self.convert_to = convert_to
        self.map_between_minus_one_and_one = map_between_minus_one_and_one
        self.flatten_xy = flatten_xy

        self.target_tensor, self.image_size = get_target_tensor(
            image_path=image_path,
            convert_to=convert_to,
            map_between_minus_one_and_one=map_between_minus_one_and_one,
            flatten_xy=True,
            also_return_image_size=True,
        )

        self.linear_space = generate_lin_space(
            image_size=self.image_size,
            flatten_xy=True,
        )

    def __len__(self):
        return self.image_size[0] * self.image_size[1]

    def __getitem__(self, idx):
        # returns pixel coortinates and pixel values
        # in the default case (X, Y) -> (R, G, B)
        # both normalized
        return self.linear_space[idx], self.target_tensor[idx]


class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, images_folder, convert_to="L"):
        self.images_folder = images_folder
        self.convert_to = convert_to
        images = os.listdir(images_folder)
        images = sorted(images, key=lambda x: int(x.split('_')[1].split('.')[0]))
        self.images = [os.path.join(images_folder, image) for image in images]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # returns frame id and frame
        # frame id is normalized between 0 and 1
        image_path = self.images[idx]
        image = load_image(image_path, convert_to=self.convert_to)
        image = image.to(torch.float32)
        image = image / 255.0

        input_tensor = idx / len(self)
        input_tensor = torch.tensor(input_tensor, dtype=torch.float32)
        input_tensor = input_tensor.unsqueeze(-1)
        return input_tensor, image

def generate_3d_linear_space(image_size=(28, 28), num_frames=10):
    # generate 3d linear space of coordinates
    # numbers will be between 0 and 1

    X = torch.linspace(0, 1, image_size[0])
    Y = torch.linspace(0, 1, image_size[1])
    Z = torch.linspace(0, 1, num_frames)
    grid = torch.stack(torch.meshgrid(X, Y, Z, indexing='ij'))
    grid = grid.permute(1, 2, 3, 0)
    # flatten the first three dimensions
    grid = grid.flatten(0, 2)
        
    return grid

class VideoPixelDataset(torch.utils.data.Dataset):
    def __init__(self, images_folder, convert_to="L"):
        print("Loading images...")
        images = sorted(os.listdir(images_folder), key=lambda x: int(x.split('_')[1].split('.')[0]))
        self.images = [os.path.join(images_folder, image) for image in images]
        self.images = [load_image(image, convert_to=convert_to) for image in self.images]
        self.images = torch.stack(self.images)
        self.images = self.images.to(torch.float32)
        self.images = self.images / 255.0
        self.num_images, self.image_x, self.image_y = self.images.shape
        self.images = self.images.flatten(0, 2)
        print("Done loading images...")
    

        print("Generating linear space...")
        self.linear_space = generate_3d_linear_space(
            image_size=(self.image_x, self.image_y),
            num_frames=self.num_images,
        )
        print("Done generating linear space...")

    def __len__(self):
        return self.num_images * self.image_x * self.image_y

    def __getitem__(self, idx):
        # it is a video dataset but we get a pixel at a time
        # so our model will input (x, y, frame_id) and output pixel value
        # everything is normalized between 0 and 1
        # so there will be num_frames * image_x * image_y samples

        input_tensor = self.linear_space[idx]
        pixel_value = self.images[image_idx, pixel_x, pixel_y]
        return input_tensor, pixel_value