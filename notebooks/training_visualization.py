# notebooks/training_visualization.py

import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# ─────────────────────────────────────────────
# PATH CONFIGURATION — SAFELY RESOLVE PROJECT ROOT
# ─────────────────────────────────────────────
# Dynamically detect your project's root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Set full file paths for CSV and image folder
LOG_CSV = os.path.join(PROJECT_ROOT, 'logs', 'training_log.csv')
SAMPLES_DIR = os.path.join(PROJECT_ROOT, 'results', 'generated_images')

# Print paths for debug purposes
print(f"[DEBUG] Resolved log path: {LOG_CSV}")
print(f"[DEBUG] Resolved samples dir: {SAMPLES_DIR}")

# ─────────────────────────────────────────────
# LOSS CURVE PLOTTING — GAN Generator/Discriminator
# ─────────────────────────────────────────────
def plot_loss_curves(log_path):
    """
    Plot GAN training loss curves from CSV file.

    Expected columns: 'Epoch', 'Generator_Loss', 'Discriminator_Loss'
    """
    if not os.path.exists(log_path):
        print(f"Log file not found at {log_path}. Skipping loss plot.")
        return

    df = pd.read_csv(log_path)

    # Validate required columns
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
# IMAGE GRID DISPLAY — GENERATED IMAGE CHECK & VISUALIZATION
# ─────────────────────────────────────────────
def show_grid(image_paths, cols=5, figsize=(12, 6)):

    # Displaying a grid of sample images from GAN output.
    # image_paths: list of full image file paths
    # cols: number of columns in the grid
    # figsize: size of matplotlib figure

    imgs_with_names = []

    for path in image_paths:
        print(f"[DEBUG] Checking image: {path}")
        if os.path.exists(path):
            try:
                img = Image.open(path).convert('RGB')
                imgs_with_names.append((img, os.path.basename(path)))
            except Exception as e:
                print(f"Could not open {path}: {e}")
        else:
            print(f"File not found: {path}")

    if not imgs_with_names:
        print("No valid images found. Skipping grid display.")
        return

    rows = int(np.ceil(len(imgs_with_names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()

    for ax, (img, name) in zip(axes, imgs_with_names):
        ax.imshow(img)
        ax.set_title(name, fontsize=8)
        ax.axis('off')

    # Turn off unused axes
    for ax in axes[len(imgs_with_names):]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

# ─────────────────────────────────────────────
# RUN VISUALIZATION PIPELINE
# ─────────────────────────────────────────────

# Plot training loss curves
plot_loss_curves(LOG_CSV)

# Define which epochs’ sample images to show
epochs_to_show = [1, 5, 10, 20, 30]
paths = [os.path.join(SAMPLES_DIR, f'epoch_{e:03d}.png') for e in epochs_to_show]

# Show image grid of selected epochs
show_grid(paths)

