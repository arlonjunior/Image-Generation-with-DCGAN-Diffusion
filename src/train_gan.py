# src/train_gan.py

import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image

from models.generator import Generator
from models.discriminator import Discriminator
from src.utils import weights_init, load_cifar10_data

# ─────────────────────────────────────────────
# ⚙️ Hyperparameters
# ─────────────────────────────────────────────
batch_size = 128
nz = 100            # Latent dimension size
ngf = 64            # Generator feature maps
ndf = 64            # Discriminator feature maps
num_epochs = 30
lr = 0.0002
beta1 = 0.5
resume = True       # Resume from checkpoint if available

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────
# 📁 Directory setup
# ─────────────────────────────────────────────
project_root = "C:/Users/arlon/PycharmProjects/DLGAI_Project"
data_path = os.path.join(project_root, "data/cifar-10-batches-py")
image_output_dir = os.path.join(project_root, "results/generated_images")
checkpoint_path = os.path.join(project_root, "models/checkpoints/checkpoint_latest.pt")
log_path = os.path.join(project_root, "logs/training_log.csv")

os.makedirs(image_output_dir, exist_ok=True)
os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
os.makedirs(os.path.dirname(log_path), exist_ok=True)

# ─────────────────────────────────────────────
# 🧠 Load and preprocess CIFAR-10 data
# ─────────────────────────────────────────────
images_np, _ = load_cifar10_data(data_path)  # Returns (N, 32, 32, 3)

# Convert to torch tensor and reshape to (N, C, H, W)
images = torch.tensor(images_np).permute(0, 3, 1, 2).float() / 255.0
images = images * 2 - 1  # Normalize to [-1, 1]

dataloader = DataLoader(TensorDataset(images), batch_size=batch_size, shuffle=True)

# ─────────────────────────────────────────────
# 🧠 Model Initialization
# ─────────────────────────────────────────────
netG = Generator(latent_dim=nz, img_channels=3, features_g=ngf).to(device)
netD = Discriminator(img_channels=3, features_d=ndf).to(device)
netG.apply(weights_init)
netD.apply(weights_init)

criterion = nn.BCELoss()
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))

# ─────────────────────────────────────────────
# ♻️ Resume from checkpoint
# ─────────────────────────────────────────────
start_epoch = 0
if resume and os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    netG.load_state_dict(checkpoint['generator_state_dict'])
    netD.load_state_dict(checkpoint['discriminator_state_dict'])
    optimizerG.load_state_dict(checkpoint['optimizer_G_state_dict'])
    optimizerD.load_state_dict(checkpoint['optimizer_D_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"✅ Resuming training from epoch {start_epoch}")

# ─────────────────────────────────────────────
# 📝 Logging CSV
# ─────────────────────────────────────────────
if not os.path.exists(log_path):
    with open(log_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Generator_Loss", "Discriminator_Loss"])

# ─────────────────────────────────────────────
# 🔁 Training Loop
# ─────────────────────────────────────────────
for epoch in range(start_epoch, num_epochs):
    g_loss_total = 0.0
    d_loss_total = 0.0

    for batch in dataloader:
        real = batch[0].to(device)
        b_size = real.size(0)

        # --- Train Discriminator ---
        netD.zero_grad()
        label_real = torch.ones(b_size, device=device)
        output_real = netD(real).view(-1)
        loss_real = criterion(output_real, label_real)

        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake = netG(noise)
        label_fake = torch.zeros(b_size, device=device)
        output_fake = netD(fake.detach()).view(-1)
        loss_fake = criterion(output_fake, label_fake)

        lossD = loss_real + loss_fake
        lossD.backward()
        optimizerD.step()
        d_loss_total += lossD.item()

        # --- Train Generator ---
        netG.zero_grad()
        label_gen = torch.ones(b_size, device=device)
        output_gen = netD(fake).view(-1)
        lossG = criterion(output_gen, label_gen)
        lossG.backward()
        optimizerG.step()
        g_loss_total += lossG.item()

    # 📝 Write losses to CSV
    with open(log_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch + 1, round(g_loss_total, 4), round(d_loss_total, 4)])

    # 🖼 Save sample image grid
    with torch.no_grad():
        sample_noise = torch.randn(64, nz, 1, 1, device=device)
        generated_samples = netG(sample_noise)
        save_image(generated_samples, os.path.join(image_output_dir, f"epoch_{epoch + 1:03d}.png"), normalize=True, nrow=8)

    # 💾 Save checkpoint
    checkpoint = {
        'epoch': epoch,
        'generator_state_dict': netG.state_dict(),
        'discriminator_state_dict': netD.state_dict(),
        'optimizer_G_state_dict': optimizerG.state_dict(),
        'optimizer_D_state_dict': optimizerD.state_dict()
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"💾 Saved checkpoint at epoch {epoch + 1} — G Loss: {g_loss_total:.4f}, D Loss: {d_loss_total:.4f}")

print("🏁 Training complete.")
