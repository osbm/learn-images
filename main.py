from learn_images import train
from learn_images import SimpleMLP
from learn_images import Fourier2DFeatures, GaussianFourierFeatures
import torch

feature_extractor = Fourier2DFeatures()

model = SimpleMLP(
    hidden_size=12,
    num_hidden_layers=4,
    init_size=feature_extractor.output_dim,
    output_size=3,
    output_activation="sigmoid"
)

model = torch.nn.Sequential(feature_extractor, model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train(
    experiment_name="fourier2dmapping",
    image_path="data/target.jpeg",
    output_folder="output_folder",
    model=model,
    optimizer=optimizer,
    max_epochs=1000,
    early_stopping_patience=50,
    save_every=1,
    disable_wandb=True,
)
