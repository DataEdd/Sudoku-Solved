"""
Custom CNN for Sudoku digit recognition.

Small architecture (~120K params) designed to be:
- Fast enough for real-time inference on 81 cells
- Accurate on printed/handwritten digits 0-9
- Simple enough to explain in a portfolio interview

Architecture:
    Input: (1, 28, 28) grayscale
    Conv(1→32, 3x3) → BN → ReLU → MaxPool(2)      -> (32, 14, 14)
    Conv(32→64, 3x3) → BN → ReLU → MaxPool(2)      -> (64, 7, 7)
    Conv(64→128, 3x3, pad=1) → BN → ReLU → AvgPool  -> (128, 1, 1)
    Flatten → Linear(128→64) → ReLU → Dropout(0.3)
    Linear(64→10)

Output: 10 classes (0=empty, 1-9=digits)
"""

import torch
import torch.nn as nn


class SudokuCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: (1, 28, 28) -> (32, 14, 14)
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2: (32, 14, 14) -> (64, 7, 7)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3: (64, 7, 7) -> (128, 1, 1)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
