from models import SimpleMLP
from PIL import Image
import os
import numpy as np
import torch
from tqdm import tqdm
from utils import generate_lin_space

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = torch.nn.MSELoss()
model = SimpleMLP()
model.to(device)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

os.makedirs("example_mnist", exist_ok=True)

target_tensor = Image.open("data/target.jpeg")
target_tensor = np.array(target_tensor)
target_tensor = torch.tensor(target_tensor)
target_tensor = target_tensor.float()
target_tensor = target_tensor / 255
target_tensor = target_tensor.unsqueeze(2)
target_tensor = target_tensor.reshape(-1, 1)
target_tensor = target_tensor.to(device)

linear_space = generate_lin_space(image_size=(28, 28))
linear_space = linear_space.to(device)
frame = 0
save_every = 20
for epoch_idx in range(1000):
    output = model(linear_space)
    loss = criterion(output, target_tensor)
    print(f"Epoch {epoch_idx} loss: {loss.item()}")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
    optimizer.step()
    optimizer.zero_grad()

    if epoch_idx % save_every == 0:
        output = output.detach().cpu().reshape(28, 28)
        output = output * 255
        output = output.to(torch.uint8)
        output = Image.fromarray(output.numpy())
        output.save(f"example_mnist/{frame}.png")
        frame += 1
