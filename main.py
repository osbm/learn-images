from learn_images.trainer import train
from learn_images.models import SimpleMLP, CorrectFourierFeatures
import torch

feature_extractor = CorrectFourierFeatures()
model = SimpleMLP(hidden_size=128, num_hidden_layers=4, init_size=feature_extractor.num_features_per_channel*2, output_size=3, output_activation="sigmoid")

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train(
    experiment_name="gaussian_fourier_mapping",
    image_path="data/target.jpeg",
    output_folder="output_folder",
    model=model,
    feature_extractor=feature_extractor,
    optimizer=optimizer,
    max_epochs=1000,
    early_stopping_patience=50,
    save_every=1,
    disable_wandb=True,
)
