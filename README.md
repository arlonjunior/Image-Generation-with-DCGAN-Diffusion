# Image-Generation-with-DCGAN-Diffusion - CIFAR-10

This project trains a Deep Convolutional GAN (DCGAN) using PyTorch on CIFAR-10 images. It includes checkpointing, metric evaluation (FID / IS), sample visualization, and logging for reproducibility and tracking.

---

## Project Workflow

**Step-by-step run order:**

| Step | Script | Description |
|------|--------|-------------|
| 1️⃣ | `convert_cifar.py` | Converts CIFAR-10 batch files into `.png` images |
| 2️⃣ | `src/train_gan.py` | Trains GAN on batch data, logs loss, saves samples/checkpoints |
| 3️⃣ | `src/evaluate.py` | Computes FID and Inception Score using saved images |
| 4️⃣ | `notebooks/metric_analysis.py` | Visualizes FID/IS scores over time |
| 5️⃣ | `notebooks/training_visualization.py` | Visualizes training losses and sample grids |
| 6️⃣ | `src/train_diffusion.py` | Trains a U-Net diffusion model |

---

## Project Structure

```
DLGAI_Project/
├── data/
│   ├── cifar-10-batches-py/         # Raw CIFAR-10 Python files
│   └── cifar10_images/              # Extracted .png image files
│
├── convert_cifar.py                 # Converts CIFAR batches into .png files
├── requirements.txt                 # Python dependencies
├── README.md                        # To be generated
│
├── src/                             # Main code and logic
│   ├── __init__.py
│   ├── evaluate.py                  # Evaluates FID and Inception Score
│   ├── inception_score.py           # Contains IS computation logic
│   ├── train_gan.py                 # DCGAN training loop
│   ├── train_diffusion.py           # U-Net Diffusion training loop
│   ├── utils.py                     # CIFAR data loading + helpers
│
├── models/                          # Model architectures
│   ├── __init__.py
│   ├── generator.py                 # DCGAN Generator
│   ├── discriminator.py             # DCGAN Discriminator
│   └── checkpoints/                 # Optional saved weights
│
├── notebooks/                       # Evaluation and visualization 
│   ├── metric_analysis.py           # Metric plotting notebook
│   ├── training_visualization.py    # Loss curves and image grid viewer  
│   ├── visualize_diffusion.py       # Visualizes diffusion simulations via plots 

│
├── results/                         # Models images saved
│   ├── generated_images/            # Output images from GAN training
│   └── diffusion_reconstructions/   # Output images from Diffusion training
│
├── logs/                            # Model logs for analysis
│   ├── training_log.csv             # Generator/Discriminator losses per epoch
│   ├── diffusion_log.csv            # Diffusion_Loss losses per epoch
│   └── metrics.csv                  # FID / IS scores per checkpoint

```

---

## Requirements

- Python 3.12
- PyTorch + torchvision
- NumPy, PIL, tqdm
- Pytorch-fid (for FID score)
- Matplotlib / pandas (for visualization notebooks)

Install all dependencies via:

`pip install -r requirements.txt`

## Checkpointing & Recovery
- After each epoch, a checkpoint is saved to models/checkpoints/checkpoint_latest.pt

- Resume training anytime by setting resume = True in train_gan.py

- Checkpoint includes model weights, optimizers, and last epoch

## Evaluation Metrics
Run:

`python src/evaluate.py`

Results are saved to `logs/metrics.csv` with timestamp and experiment label.

## Visualization
View sample images from `results/generated_images/`

Launch `notebooks` to graph training curves and evaluate model quality

## Sharing
To share progress:

- Send `checkpoint_latest.pt` for reproducibility

- Include `training_log.csv` and `metrics.csv` for performance metrics

- Share visuals from `results/` or screenshots of `notebook` plots

## TODOs & Extensions
- [ ] Add conditional GAN with class labels

- [ ] Implement style-based generator (e.g. StyleGAN lite)

- [ ] Push best models to HuggingFace

## Credits
Built and published by Arlon Junior Moleka using PyTorch, CIFAR-10.