from learn_images.trainer import train
from learn_images.models import SimpleMLP, SkipConnectionsMLP

# model = SimpleMLP(hidden_size=100, num_hidden_layers=7, init_size=2, output_size=3)
# model = SkipConnectionsMLP(hidden_size=100, num_hidden_layers=7, init_size=2, output_size=3)


train(
    image_path="data/target.jpeg",
    output_folder="output_folder",
    model=model,
    max_epochs=1000,
    early_stopping_patience=50,
    save_every=5
)