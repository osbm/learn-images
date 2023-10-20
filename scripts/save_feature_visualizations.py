from learn_images import generate_lin_space, save_features_to_folder
from learn_images import GaussianFourierFeatures, FourierFeatures, Fourier2DFeatures
from pathlib import Path

visualizations_folder = Path("data/feature_visualizations")

linear_space = generate_lin_space(image_size=(720, 1280))
save_features_to_folder(linear_space, output_folder=visualizations_folder / "linear_space")

fourier_features = FourierFeatures(
    fourier_order=4,
    exponential_increase=True,
    multiply_by_2pi=True,
)(linear_space)
save_features_to_folder(fourier_features, output_folder=visualizations_folder / "fourier_features")

gaussian_fourier_mapping = GaussianFourierFeatures(
    mapping_size=8,
    scale=10,
    input_vector_size=2
)(linear_space)
save_features_to_folder(gaussian_fourier_mapping, output_folder=visualizations_folder / "gaussian_fourier_features")

fourier2d_features = Fourier2DFeatures(
    fourier_order=4,
)(linear_space)
save_features_to_folder(fourier2d_features, output_folder=visualizations_folder / "fourier2d_features")
