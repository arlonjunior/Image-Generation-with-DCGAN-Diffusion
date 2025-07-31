import torch
from src.inception_score import get_inception_score

# Path to generated images
generated_path = 'C:/Users/arlon/PycharmProjects/DLGAI_Project/results/generated_images'

# Run Inception Score
is_score, is_std = get_inception_score(generated_path)
print(f"Inception Score: {is_score:.2f} ± {is_std:.2f}")