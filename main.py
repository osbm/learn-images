from learn_images.trainer import train
from learn_images.models import SimpleMLP, SkipConnectionsMLP, Fourier2DFeatures, FourierFeatues, PadeFeatures

from torch import nn
import torch
# model = SimpleMLP(hidden_size=100, num_hidden_layers=7, init_size=2, output_size=3)

# feature_extractor = FourierFeatues(fourier_order=4)
# feature_extractor = Fourier2DFeatures(fourier_order=4)
# create a tesnor with 2,2 shape
# tensor = torch.tensor([[1,2]]).float()
# print(feature_extractor(tensor).shape)
model = SimpleMLP(
    hidden_size=100,
    num_hidden_layers=7,
    init_size=2,
    output_size=3
)
# model = nn.Sequential(feature_extractor, model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

train(
    experiment_name="debug2",
    image_path="data/target.jpeg",
    output_folder="output_folder",
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    max_epochs=3000,
    early_stopping_patience=50,
    save_every=5
)