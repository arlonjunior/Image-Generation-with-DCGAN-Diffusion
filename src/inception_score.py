# src/inception_score.py

import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights

# ─────────────────────────────────────────────
# 🎯 Inception Score Calculation
# ─────────────────────────────────────────────
def get_inception_score(img_folder, splits=10, device=None):
    """
    Calculates the Inception Score for generated images.

    Args:
        img_folder (str): Path to folder containing images.
        splits (int): Number of splits for stability.
        device (str): 'cuda' or 'cpu'. Auto-detected if None.

    Returns:
        Tuple[float, float]: Mean and std of Inception Score.
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    weights = Inception_V3_Weights.DEFAULT
    model = inception_v3(weights=weights, transform_input=False).to(device)
    model.eval()

    # ✅ FIXED normalization (no .meta access)
    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet standard
                             std=[0.229, 0.224, 0.225])
    ])

    images = []
    for img_name in tqdm(os.listdir(img_folder), desc='🔍 Loading images'):
        path = os.path.join(img_folder, img_name)
        if os.path.isfile(path):
            try:
                img = Image.open(path).convert('RGB')
                img = preprocess(img).unsqueeze(0)
                images.append(img)
            except Exception as e:
                print(f"⚠️ Skipping {img_name}: {e}")

    if not images:
        raise ValueError("🚫 No valid images found in folder.")

    images = torch.cat(images, dim=0).to(device)

    with torch.no_grad():
        preds = F.softmax(model(images), dim=1).cpu().numpy()

    split_scores = []
    for k in range(splits):
        part = preds[k * (len(preds) // splits): (k + 1) * (len(preds) // splits)]
        py = np.mean(part, axis=0)
        scores = [np.sum(p * (np.log(p + 1e-10) - np.log(py + 1e-10))) for p in part]
        split_scores.append(np.exp(np.mean(scores)))

    return float(np.mean(split_scores)), float(np.std(split_scores))
