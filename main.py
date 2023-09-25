from learn_images.trainer import train
from learn_images.models import SimpleMLP, SkipConnectionsMLP, Fourier2DFeatures, FourierFeatues, PadeFeatures

from torch import nn
# import torch
# model = SimpleMLP(hidden_size=100, num_hidden_layers=7, init_size=2, output_size=3)

# feature_extractor = FourierFeatues(fourier_order=4)
feature_extractor = PadeFeatures(pade_order=4)
# create a tesnor with 2,2 shape
# tensor = torch.tensor([[1,2]]).float()
# print(feature_extractor(tensor).shape)
model = SimpleMLP(
    hidden_size=100,
    num_hidden_layers=7,
    init_size=feature_extractor.output_shape,
    output_size=3
)
model = nn.Sequential(feature_extractor, model)


train(
    experiment_name="debug",
    image_path="data/target.jpeg",
    output_folder="output_folder",
    model=model,
    max_epochs=3000,
    early_stopping_patience=50,
    save_every=5
)