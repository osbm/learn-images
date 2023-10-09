from learn_images.utils import generate_lin_space, get_target_tensor
from learn_images.models import CorrectFourierFeatures
import os
import matplotlib.image

target_tensor = get_target_tensor(file_path="data/target.jpeg")
linear_space = generate_lin_space(image_size=target_tensor.shape)
feature_extractor = CorrectFourierFeatures()
linear_space_mapped = feature_extractor(linear_space)


print("Deleting the linear space visualizations of previous runs.")
for file in os.listdir("data"):
    if file.startswith("linear_space"):
        os.remove(os.path.join("data", file))

print("Saving the linear space.")
for i in range(linear_space.shape[-1]):
    matplotlib.image.imsave(
        f"data/linear_space_input_{i}.png",
        linear_space[:, :, i].numpy(),
        cmap="gray",
    )

print("Saving the linear space mapped.")
for i in range(linear_space_mapped.shape[-2]): # number of input features
    for j in range(linear_space_mapped.shape[-1]): # number of extracted features
        matplotlib.image.imsave(
            f"data/linear_space_mapped_input_{i}_feature_{j}.png",
            linear_space_mapped[:, :, i, j].numpy(),
            cmap="gray",
        )
