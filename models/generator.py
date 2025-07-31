# generator.py

import torch
import torch.nn as nn

class Generator(nn.Module):
    """
    DCGAN-style Generator for 32×32 images (CIFAR-10).
    Generates an RGB image from latent noise vector.

    Args:
        latent_dim (int): Size of the input noise vector (e.g. 100)
        img_channels (int): Number of output image channels (3 for RGB)
        features_g (int): Base feature map size (e.g. 64)
    """

    def __init__(self, latent_dim=100, img_channels=3, features_g=64):
        super().__init__()
        self.latent_dim = latent_dim

        # Define generator layers
        self.net = nn.Sequential(
            # Input Z: (B, latent_dim, 1, 1) → Output: (B, features_g*4, 4, 4)
            self._block(latent_dim, features_g * 4, 4, 1, 0),

            # Output: (B, features_g*2, 8, 8)
            self._block(features_g * 4, features_g * 2, 4, 2, 1),

            # Output: (B, features_g, 16, 16)
            self._block(features_g * 2, features_g, 4, 2, 1),

            # Output: (B, img_channels, 32, 32)
            nn.ConvTranspose2d(features_g, img_channels, 4, 2, 1),
            nn.Tanh()
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        """Defines one transposed convolution block: ConvTranspose2d → BatchNorm → ReLU"""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward method to map latent vector z to generated image.
        Accepts z shaped (B, latent_dim) or (B, latent_dim, 1, 1)
        Returns image shaped (B, img_channels, 32, 32)
        """
        if z.dim() == 2:
            z = z.view(z.size(0), self.latent_dim, 1, 1)
        return self.net(z)