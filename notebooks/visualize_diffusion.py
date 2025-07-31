# ─────────────────────────────────────────────
# 📊 Diffusion Training Visualization
# ─────────────────────────────────────────────
# Combines:
#   1. Loss curve plotting from CSV log
#   2. Epoch-wise reconstruction viewing
# ─────────────────────────────────────────────

import os
import csv
from PIL import Image
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# 📁 Safe Path Resolution
# ─────────────────────────────────────────────

# Detect script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Go up one level to reach the project root
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

# Absolute paths to log file and recon directory
log_path = os.path.join(PROJECT_ROOT, "logs", "diffusion_log.csv")
recon_dir = os.path.join(PROJECT_ROOT, "results", "diffusion_reconstructions")

print(f"[DEBUG] Log path: {log_path}")
print(f"[DEBUG] Recon path: {recon_dir}")

# ─────────────────────────────────────────────
# 📈 Plot Loss Curve from CSV
# ─────────────────────────────────────────────

def plot_loss_curve():
    if not os.path.exists(log_path):
        print(f"❌ Log file not found: {log_path}")
        return

    epochs, losses = [], []

    # Read the diffusion CSV log
    with open(log_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["Epoch"]))
            losses.append(float(row["Diffusion_Loss"]))

    # Plot training loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, losses, marker='o', linestyle='-', color='teal', label='Diffusion Loss')
    plt.title("Diffusion Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()

    # Optional: save to disk
    save_path = os.path.join(PROJECT_ROOT, "results", "diffusion_loss_curve.png")
    plt.savefig(save_path)
    print(f"✅ Saved loss plot to: {save_path}")
    plt.show()

# ─────────────────────────────────────────────
# 🖼 View Reconstruction Images Across Epochs
# ─────────────────────────────────────────────

def view_reconstructions(epochs_to_view):
    for epoch in epochs_to_view:
        filename = f"epoch_{epoch:03d}.png"
        image_path = os.path.join(recon_dir, filename)

        if os.path.exists(image_path):
            img = Image.open(image_path)
            plt.figure(figsize=(10, 3))
            plt.imshow(img)
            plt.axis('off')
            plt.title(f"Diffusion Reconstructions — Epoch {epoch}")
            plt.show()
        else:
            print(f"⚠️ Epoch {epoch} image not found at: {image_path}")

# ─────────────────────────────────────────────
# 🚀 Run Visualizations
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("📊 Visualizing Diffusion Logs...")
    plot_loss_curve()

    # 🧮 Epochs you want to inspect (customize anytime)
    desired_epochs = [1, 3, 5, 7, 10]
    view_reconstructions(desired_epochs)
