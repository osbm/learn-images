import torch
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
from .utils import generate_lin_space
import wandb

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def train(
    experiment_name="debug",
    image_path=None,
    output_folder=None,
    model=None,
    optimizer=None,
    scheduler=None,
    max_epochs=1000,
    early_stopping_patience=50,
    save_every=5,
    seed=42
):
    set_seed(seed)
    model_name = model.__class__.__name__
    if model_name == "Sequential":
        model_name = f"{model[0].__class__.__name__}_{model[1].__class__.__name__}_order_{model[0].fourier_order}"
        num_hidden_layers = model[1].num_hidden_layers
        hidden_size = model[1].hidden_size
    else:
        num_hidden_layers = model.num_hidden_layers
        hidden_size = model.hidden_size


    wandb.init(
        project="learn-images",
        name=experiment_name,
        config={
            "max_epochs": max_epochs,
            "early_stopping_patience": early_stopping_patience,
            "save_every": save_every,
            "seed": seed,
            "model_name": model_name,
            "model_num_hidden_layers": num_hidden_layers,
            "model_hidden_size": hidden_size,
            "optimizer": optimizer.__class__.__name__,
            "optimizer_config": optimizer.state_dict()["param_groups"],
            "scheduler": scheduler.__class__.__name__,
            "scheduler_config": scheduler.state_dict() if scheduler else None,
        }
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.MSELoss()
    model.to(device)

    os.makedirs(output_folder, exist_ok=True)

    target_tensor = Image.open(image_path)
    target_tensor = np.array(target_tensor)
    target_tensor = torch.tensor(target_tensor)
    image_size = target_tensor.shape
    target_tensor = target_tensor.float()
    target_tensor = target_tensor / 255
    target_tensor = target_tensor.reshape(-1, image_size[2])
    target_tensor = target_tensor.to(device)

    linear_space = generate_lin_space(image_size=image_size)
    linear_space = linear_space.to(device)
    frame = 0

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
        wandb.log({"loss": loss, "learning_rate": optimizer.param_groups[0]['lr']})
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
        optimizer.step()
        optimizer.zero_grad()
        if scheduler:
            scheduler.step()

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
    # save all losses to a txt file
    # with open("output_folder/losses.txt", "w") as f:
    #     for loss in all_losses:
    #         f.write(f"{loss}\n")
        