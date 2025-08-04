# convert_cifar.py

import os
import pickle
import numpy as np
from PIL import Image

CIFAR_DIR = 'data/cifar-10-batches-py'
OUT_DIR = 'data/cifar10_images'
os.makedirs(OUT_DIR, exist_ok=True)

def unpickle(file):
    with open(file, 'rb') as fo:
        return pickle.load(fo, encoding='bytes')

def convert_batch(batch_name):
    batch = unpickle(os.path.join(CIFAR_DIR, batch_name))
    for i, img in enumerate(batch[b'data']):
        r, g, b = img[0:1024], img[1024:2048], img[2048:]
        r = r.reshape(32, 32)
        g = g.reshape(32, 32)
        b = b.reshape(32, 32)
        stacked_img = np.stack([r, g, b], axis=2)
        im = Image.fromarray(stacked_img)
        im.save(os.path.join(OUT_DIR, f"{batch_name}_{i}.png"))

# Run for each data batch
for batch_file in ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']:
    convert_batch(batch_file)

print("CIFAR-10 images exported to:", OUT_DIR)

