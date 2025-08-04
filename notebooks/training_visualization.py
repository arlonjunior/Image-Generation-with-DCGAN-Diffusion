# notebooks/training_visualization.py

import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# ─────────────────────────────────────────────
# PATH CONFIGURATION — SAFE PROJECT RESOLUTION
# ─────────────────────────────────────────────
# Dynamically locate your project's root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Full file paths for training logs and GAN outputs
LOG_CSV = os.path.join(PROJECT_ROOT, 'logs', 'training_log.csv')
SAMPLES_DIR = os.path.join(PROJECT_ROOT, 'results', 'generated_images')

# Debug output to confirm path resolution
print(f"[DEBUG] Resolved log path: {LOG_CSV}")
print(f"[DEBUG] Resolved samples dir: {SAMPLES_DIR}")

# ─────────────────────────────────────────────
# LOSS CURVE PLOTTING FOR GAN
# ─────────────────────────────────────────────
def plot_loss_curves(log_path):
    """
    Plot Generator and Discriminator loss curves from training log CSV.
    Expected columns: 'Epoch', 'Generator_Loss', 'Discriminator_Loss'
    """
    if not os.path.exists(log_path):
        print(f"Log file not found at {log_path}. Skipping loss plot.")
        return

    df = pd.read_csv(log_path)

    # Check for required columns
    if {'Epoch', 'Generator_Loss', 'Discriminator_Loss'}.issubset(df.columns):
        plt.figure(figsize=(8, 5))
        plt.plot(df['Epoch'], df['Generator_Loss'], label='Generator Loss', color='blue')
        plt.plot(df['Epoch'], df['Discriminator_Loss'], label='Discriminator Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('GAN Training Loss Over Epochs')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print("CSV missing expected columns: 'Epoch', 'Generator_Loss', 'Discriminator_Loss'")

# ─────────────────────────────────────────────
# INDIVIDUAL IMAGE DISPLAY FOR SELECTED EPOCHS
# ─────────────────────────────────────────────
def show_individual_epochs(epoch_list, samples_dir):

    # Display of each GAN-generated image for each epoch

    for epoch in epoch_list:
        file_name = f"epoch_{epoch:03d}.png"
        image_path = os.path.join(samples_dir, file_name)
        print(f"[DEBUG] Loading image for epoch {epoch}: {image_path}")

        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue

        try:
            img = Image.open(image_path).convert('RGB')
            plt.figure(figsize=(6, 6))
            plt.imshow(img)
            plt.title(f"Generated Image — Epoch {epoch}")
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Failed to open {image_path}: {e}")

# ─────────────────────────────────────────────
# EXECUTE VISUALIZATION PIPELINE
# ─────────────────────────────────────────────

# Step 1: Plot training loss curves
plot_loss_curves(LOG_CSV)

# Step 2: Display individual sample images for selected epochs
epochs_to_show = [1, 5, 10, 20, 30]
show_individual_epochs(epochs_to_show, SAMPLES_DIR)


