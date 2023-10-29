import torch
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
from .utils import generate_lin_space, get_target_tensor, set_seed
import wandb
from .datasets import ImageDataset, VideoDataset

def train_image_model(
    experiment_name="debug",
    image_path=None,
    output_folder=None,
    model=None,
    feature_extractor=None,
    optimizer=None,
    scheduler=None,
    max_epochs=1000,
    early_stopping_patience=50,
    save_every=5,
    seed=42,
    disable_wandb=False,
):
    set_seed(seed)

    wandb.init(
        project="learn-images",
        name=experiment_name,
        config={
            "max_epochs": max_epochs,
            "early_stopping_patience": early_stopping_patience,
            "save_every": save_every,
            "seed": seed,
            "model_name": model.__class__.__name__,
            "optimizer": optimizer.__class__.__name__,
            "optimizer_config": optimizer.state_dict()["param_groups"],
            "scheduler": scheduler.__class__.__name__,
            "scheduler_config": scheduler.state_dict() if scheduler else None,
        },
        mode="disabled" if disable_wandb else None,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.MSELoss()

    os.makedirs(output_folder, exist_ok=True)

    target_tensor, image_size = get_target_tensor(file_path=image_path, also_return_image_size=True)

    linear_space = generate_lin_space(image_size=image_size)
    frame = 0

    target_tensor = target_tensor.flatten(0, 1)
    linear_space = linear_space.flatten(0, 1)

    best_loss = float('inf')
    consecutive_epochs_no_improvement = 0
    max_consecutive_epochs_no_improvement = early_stopping_patience  # Set the threshold for early stopping

    # move stuff to device
    target_tensor = target_tensor.to(device)
    linear_space = linear_space.to(device)
    model.to(device)

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
        wandb.log({"loss": loss, "learning_rate": optimizer.param_groups[0]['lr']})
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
        optimizer.step()
        optimizer.zero_grad()
        if scheduler:
            scheduler.step(loss)

        if epoch_idx % save_every == 0:
            output = output.detach().cpu().reshape(image_size)
            output = output * 255
            output = output.to(torch.uint8)
            output = Image.fromarray(output.numpy())
            output.save(f"output_folder/{frame}.png")
            wandb.log({"output": wandb.Image(output)})
            print(f"Saved frame {frame}")
            frame += 1

        if consecutive_epochs_no_improvement >= max_consecutive_epochs_no_improvement:
            print(f"Stopping early as loss hasn't improved for {max_consecutive_epochs_no_improvement} consecutive epochs.")
            break

    wandb.finish()


def train_video_model(
    experiment_name="debug",
    frames_folder_path=None,
    output_folder=None,
    model=None,
    optimizer=None,
    scheduler=None,
    max_epochs=1000,
    early_stopping_patience=50,
    save_every=5,
    seed=42,
    batch_size=32,
    disable_wandb=False,
):
    set_seed(seed)

    wandb.init(
        project="learn-images",
        name=experiment_name,
        config={
            "max_epochs": max_epochs,
            "early_stopping_patience": early_stopping_patience,
            "save_every": save_every,
            "seed": seed,
            "model_name": model.__class__.__name__,
            "optimizer": optimizer.__class__.__name__,
            "optimizer_config": optimizer.state_dict()["param_groups"],
            "scheduler": scheduler.__class__.__name__,
            "scheduler_config": scheduler.state_dict() if scheduler else None,
        },
        mode="disabled" if disable_wandb else None,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # criterion = torch.nn.MSELoss()
    criterion = torch.nn.BCELoss()


    os.makedirs(output_folder, exist_ok=True)

    dataset = VideoDataset(images_folder=frames_folder_path, convert_to="L")
    num_images = len(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    best_loss = float('inf')
    consecutive_epochs_no_improvement = 0
    max_consecutive_epochs_no_improvement = early_stopping_patience  # Set the threshold for early stopping

    # move stuff to device
    model.to(device)

    for epoch_idx in range(max_epochs):
        losses = []
        for x, y in tqdm(dataloader):
            x, y = x.to(device), y.to(device)
            output = model(x)
            output = output.squeeze(1) # remove channel dimension
            loss = criterion(output, y)
            loss.backward()
            loss = loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
            optimizer.step()
            optimizer.zero_grad()
            if scheduler:
                scheduler.step(loss)

            losses.append(loss)
            wandb.log({"loss": loss, "learning_rate": optimizer.param_groups[0]['lr']})
            
        epoch_loss = np.mean(losses)
        print(f"Epoch {epoch_idx} loss: {epoch_loss}")
        wandb.log({"epoch_loss": epoch_loss})

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            consecutive_epochs_no_improvement = 0
        else:
            consecutive_epochs_no_improvement += 1
        
        if epoch_idx % save_every == 0 and epoch_idx > 0 and False:
            linear_space = torch.linspace(0, 1, num_images)
            linear_space = linear_space.to(device)
            # split into batches
            linear_space = torch.split(linear_space, batch_size)
            os.makedirs(f"{output_folder}/{epoch_idx}_save", exist_ok=True)
            for idx, batch in enumerate(linear_space):
                output = model(batch)
                output = output.detach().cpu().squeeze(1)
                output = output * 255
                output = output.to(torch.uint8)
                for frame_idx, frame in enumerate(output):
                    frame = Image.fromarray(frame.numpy())
                    frame.save(f"{output_folder}/{epoch_idx}_save/{idx * batch_size + frame_idx}.png")


        if consecutive_epochs_no_improvement >= max_consecutive_epochs_no_improvement:
            print(f"Stopping early as loss hasn't improved for {max_consecutive_epochs_no_improvement} consecutive epochs.")
            break

    wandb.finish()
