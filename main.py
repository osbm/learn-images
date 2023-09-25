from learn_images.trainer import train
from learn_images.models import SimpleMLP, SkipConnectionsMLP, Fourier2DFeatures, FourierFeatues, PadeFeatures

from torch import nn
import torch
# model = SimpleMLP(hidden_size=100, num_hidden_layers=7, init_size=2, output_size=3)

# feature_extractor = FourierFeatues(fourier_order=4)
feature_extractor = Fourier2DFeatures(fourier_order=4)
model = SkipConnectionsMLP(
    hidden_size=10,
    num_hidden_layers=7,
#     init_size=2,
    init_size=feature_extractor.output_shape,
    output_size=3
)
model = nn.Sequential(feature_extractor, model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

train(
    experiment_name="fourier2dskipconn",
    image_path="data/target.jpeg",
    output_folder="output_folder",
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    max_epochs=3000,
    early_stopping_patience=50,
    save_every=5,
    disable_wandb=False,
)
