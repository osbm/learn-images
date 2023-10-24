from learn_images import train_video_model, GaussianFourierFeatures, FrameGeneratorCNN
import torch
from torch import nn

feature_extractor = GaussianFourierFeatures(mapping_size=256, scale=10, input_vector_size=1)
model = FrameGeneratorCNN(input_size=feature_extractor.output_dim)

model = nn.Sequential(feature_extractor, model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_video_model(
    experiment_name="video_experiment1",
    frames_folder_path="data/bad_apple_frames",
    output_folder="output_folder",
    model=model,
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
    scheduler=None,
    max_epochs=1000,
    early_stopping_patience=50,
    save_every=5,
    seed=42,
    batch_size=2,
    disable_wandb=True,
)