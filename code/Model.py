# Model.py

import torch.nn as nn


class MC_CNN(nn.Module):
    def __init__(self, dropout_rate=0.02):
        super(MC_CNN, self).__init__()

        self.conv_branch = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=1)
        )
         
    def forward(self, region_input):
        """
        region_input: (B, 3, 320, 416)
        """
        out = self.conv_branch(region_input)
        return out.squeeze(1)  # 输出形状为 (B, 320, 416)

