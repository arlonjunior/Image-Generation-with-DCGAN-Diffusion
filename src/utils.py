# src/utils.py

import os
import pickle
import numpy as np
import torch.nn as nn


# ─────────────────────────────────────────────
# CIFAR-10 Data Loading Functions
# ─────────────────────────────────────────────

def load_cifar10_batch(filepath):
    # Loading a single CIFAR-10 data batch from the file path.
    # Path to the CIFAR-10 batch file.
    # Return the tuple of images and labels.

    with open(filepath, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    data = batch[b'data']
    labels = batch[b'labels']
    images = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return images, labels


def load_cifar10_data(data_dir):

    # Loads all CIFAR-10 training data from data batches 1–5.
    # Directory containing CIFAR-10 batch files.
    # Returns Numpy arrays of images and labels.

    all_images, all_labels = [], []
    for i in range(1, 6):
        file = os.path.join(data_dir, f'data_batch_{i}')
        imgs, lbls = load_cifar10_batch(file)
        all_images.append(imgs)
        all_labels.extend(lbls)
    x = np.concatenate(all_images, axis=0)
    y = np.array(all_labels)
    return x, y


# ─────────────────────────────────────────────
# GAN Weight Initialization Function
# ─────────────────────────────────────────────

def weights_init(m):

    # Applies custom weight initialization to layers in a model.

    # Conv layers: Normal distribution (mean=0.0, std=0.02)
    # BatchNorm layers: Normal weights (mean=1.0, std=0.02), bias=0
    # m (nn.Module): Layer to initialize.

    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
