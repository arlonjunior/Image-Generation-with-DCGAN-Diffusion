# discriminator.py

import torch
import torch.nn as nn

class Discriminator(nn.Module):
    """
    DCGAN-style Discriminator for 32×32 images.
    """

    def __init__(self, img_channels=3, features_d=64):
        super().__init__()
        self.net = nn.Sequential(
            # input: (B, img_channels, 32, 32)
            nn.Conv2d(img_channels, features_d, kernel_size=4, stride=2, padding=1, bias=False),  # 16×16
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features_d, features_d * 2, kernel_size=4, stride=2, padding=1, bias=False),  # 8×8
            nn.BatchNorm2d(features_d * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features_d * 2, features_d * 4, kernel_size=4, stride=2, padding=1, bias=False),  # 4×4
            nn.BatchNorm2d(features_d * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features_d * 4, 1, kernel_size=4, stride=1, padding=0, bias=False),  # 1×1
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.net(img).view(-1, 1)  # flatten to (B, 1)

