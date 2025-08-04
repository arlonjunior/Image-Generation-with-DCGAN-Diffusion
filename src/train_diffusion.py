# src/train_diffusion.py

import os
import csv
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image
from tqdm import tqdm

from src.utils import load_cifar10_data  # Adjust if needed

# ─────────────────────────────────────────────
# Directory & File Setup
# ─────────────────────────────────────────────

project_root = "C:/Users/arlon/PycharmProjects/DLGAI_Project"
data_path = os.path.join(project_root, "data/cifar-10-batches-py")
checkpoint_path = os.path.join(project_root, "models/checkpoints/diffusion_latest.pt")
log_path = os.path.join(project_root, "logs/diffusion_log.csv")
recon_dir = os.path.join(project_root, "results/diffusion_reconstructions")

# Ensure folders exist
os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
os.makedirs(os.path.dirname(log_path), exist_ok=True)
os.makedirs(recon_dir, exist_ok=True)

# ─────────────────────────────────────────────
# Lightweight U-Net Backbone
# ─────────────────────────────────────────────

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encode = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU()
        )
        self.decode = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1)
        )

    def forward(self, x, t):  # timestep t is unused in this basic version
        return self.decode(self.encode(x))

# ─────────────────────────────────────────────
# Load and Preprocess CIFAR-10
# ─────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

images_np, _ = load_cifar10_data(data_path)
images = torch.tensor(images_np).permute(0, 3, 1, 2).float() / 255.0
images = images * 2 - 1  # scale to [-1, 1]
dataloader = DataLoader(TensorDataset(images), batch_size=64, shuffle=True)

# ─────────────────────────────────────────────
# Initialize Model, Optimizer, Loss
# ─────────────────────────────────────────────

model = UNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()
timesteps = 1000  # max timestep index
start_epoch = 0
num_epochs = 10

# ─────────────────────────────────────────────
# Resume from Checkpoint (optional)
# ─────────────────────────────────────────────

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resumed training from epoch {start_epoch}")

# ─────────────────────────────────────────────
# Create CSV Log File if Missing
# ─────────────────────────────────────────────

if not os.path.exists(log_path):
    with open(log_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Diffusion_Loss"])

# ─────────────────────────────────────────────
# Training Loop with Reconstructions
# ─────────────────────────────────────────────

for epoch in range(start_epoch, num_epochs):
    total_loss = 0.0

    for batch in tqdm(dataloader, desc=f"🌀 Epoch {epoch}"):
        clean = batch[0].to(device)
        noise = torch.randn_like(clean)
        t = torch.randint(0, timesteps, (clean.size(0),), device=device)

        # Forward diffusion: add noise
        noisy = clean + 0.1 * noise

        # Reverse step: predict noise
        predicted = model(noisy, t)
        loss = criterion(predicted, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = round(total_loss, 4)
    print(f"Epoch {epoch} — Avg Loss: {avg_loss}")

    # Append to CSV log
    with open(log_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch + 1, avg_loss])

    # Save reconstructions (first 8 samples)
    with torch.no_grad():
        test_clean = clean[:8]
        test_noisy = noisy[:8]
        test_predicted = model(test_noisy, t[:8])

        # Reconstruct clean image from predicted noise
        reconstructed = test_noisy - test_predicted
        grid = torch.cat([test_clean, test_noisy, reconstructed], dim=0)
        save_image(grid, os.path.join(recon_dir, f"epoch_{epoch + 1:03d}.png"),
                   normalize=True, nrow=8)

    # Save checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, checkpoint_path)

    print(f"Saved checkpoint and reconstruction for epoch {epoch + 1}")

print("Diffusion training complete.")
