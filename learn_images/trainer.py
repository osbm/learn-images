import torch
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
from .utils import generate_lin_space


def train(image_path=None, output_folder=None, model=None, max_epochs=1000, early_stopping_patience=50, save_every=5):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.MSELoss()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    os.makedirs(output_folder, exist_ok=True)

    target_tensor = Image.open(image_path)
    target_tensor = np.array(target_tensor)
    target_tensor = torch.tensor(target_tensor)
    image_size = target_tensor.shape
    target_tensor = target_tensor.float()
    target_tensor = target_tensor / 255
    target_tensor = target_tensor.reshape(-1, image_size[2])
    target_tensor = target_tensor.to(device)

    # print(target_tensor.shape)
    linear_space = generate_lin_space(image_size=image_size)
    linear_space = linear_space.to(device)
    frame = 0

    all_losses = []
    best_loss = float('inf')
    consecutive_epochs_no_improvement = 0
    max_consecutive_epochs_no_improvement = early_stopping_patience  # Set the threshold for early stopping

    for epoch_idx in range(max_epochs):
        output = model(linear_space)
        loss = criterion(output, target_tensor)
        loss.backward()
        loss = loss.item()

        if loss < best_loss:
            best_loss = loss
            consecutive_epochs_no_improvement = 0
        else:
            consecutive_epochs_no_improvement += 1


        print(f"Epoch {epoch_idx} loss: {loss}")
        all_losses.append(loss)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
        optimizer.step()
        optimizer.zero_grad()
        # scheduler.step()

        if epoch_idx % save_every == 0:
            output = output.detach().cpu().reshape(image_size)
            output = output * 255
            output = output.to(torch.uint8)
            output = Image.fromarray(output.numpy())
            output.save(f"output_folder/{frame}.png")
            print(f"Saved frame {frame}")
            frame += 1

        if consecutive_epochs_no_improvement >= max_consecutive_epochs_no_improvement:
            print(f"Stopping early as loss hasn't improved for {max_consecutive_epochs_no_improvement} consecutive epochs.")
            break


    # save all losses to a txt file
    with open("output_folder/losses.txt", "w") as f:
        for loss in all_losses:
            f.write(f"{loss}\n")
        