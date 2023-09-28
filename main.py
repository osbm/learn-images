from learn_images.trainer import train
from learn_images.models import FourierFeatures, SimpleMLP, GaussianFourierMapping
from learn_images.utils import generate_lin_space
from torch import nn
import torch
# import matplotlib.image
# from PIL import Image
# import numpy as np

feature_extractor = GaussianFourierMapping()
model = SimpleMLP(hidden_size=256, num_hidden_layers=4, init_size=512, output_size=3, output_activation="sigmoid")

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

train(
    experiment_name="gaussian_fourier_mapping",
    image_path="data/target.jpeg",
    output_folder="output_folder",
    model=model,
    feature_extractor=feature_extractor,
    optimizer=optimizer,
    scheduler=scheduler,
    max_epochs=1000,
    early_stopping_patience=50,
    save_every=1,
    disable_wandb=False,
)
