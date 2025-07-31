# src/evaluate.py

import torch
from pytorch_fid import fid_score
from src.inception_score import get_inception_score
import csv
import os
from datetime import datetime

# ─────────────────────────────────────────────
# 📊 Metrics CSV Logger
# ─────────────────────────────────────────────
def log_metrics(metrics_dict, csv_path, experiment_id='baseline'):
    metrics_dict['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    metrics_dict['experiment'] = experiment_id
    write_header = not os.path.exists(csv_path)

    with open(csv_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics_dict.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(metrics_dict)

# ─────────────────────────────────────────────
# 🧪 GAN Evaluation Runner
# ─────────────────────────────────────────────
def evaluate_results():
    project_root = 'C:/Users/arlon/PycharmProjects/DLGAI_Project'
    generated_path = os.path.join(project_root, 'results', 'generated_images')
    real_path = os.path.join(project_root, 'data', 'cifar10_images')
    log_path = os.path.join(project_root, 'logs', 'metrics.csv')

    # FID Score
    fid = fid_score.calculate_fid_given_paths(
        [generated_path, real_path],
        batch_size=50,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        dims=2048
    )
    print(f"✅ FID Score: {fid:.2f}")

    # Inception Score
    is_score, is_std = get_inception_score(generated_path)
    print(f"✅ Inception Score: {is_score:.2f} ± {is_std:.2f}")

    # Save metrics
    results = {
        'fid_score': round(fid, 2),
        'inception_score': round(is_score, 2),
        'inception_std': round(is_std, 2)
    }
    log_metrics(results, csv_path=log_path, experiment_id='DCGAN_final')

# 🚀 Entry
if __name__ == "__main__":
    evaluate_results()


