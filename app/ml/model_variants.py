"""
Configurable SudokuCNN variants for architecture ablation experiments.

The production model in ``app/ml/model.py`` is a fixed-shape 102K-parameter
CNN. Its design (depth, channel counts, dropout rate) was chosen after
running the 3×3×3 ablation in ``evaluation/ablation.py`` — see
``evaluation/ablation_results.json`` for the full results and
``notebooks/03_ocr.ipynb`` ("Architecture ablation") for the analysis.

This module exists so that ``ablation.py`` can instantiate arbitrary
``(depth, channels, dropout)`` variants of the same family without
touching the production ``SudokuCNN`` class.

Example:
    The production model is exactly equivalent to

        SudokuCNNAblation(
            depth=3,
            channels=[32, 64, 128],
            dropout=0.3,
            hidden_fc=64,
        )

    and produces the same 102,026 parameters.
"""

from typing import List

import torch
import torch.nn as nn


class SudokuCNNAblation(nn.Module):
    """Parameterised CNN used by the architecture ablation study.

    The architecture family is a stack of ``depth`` Conv-BN-ReLU blocks
    followed by a small FC classifier head. The first ``depth - 1``
    blocks reduce spatial dimensions with ``MaxPool(2)``; the final
    block collapses to ``(1, 1)`` with ``AdaptiveAvgPool(1)``. All
    convolutions use ``3×3`` kernels with ``padding=1``.

    Input shape: ``(N, 1, 28, 28)`` grayscale.
    Output shape: ``(N, num_classes)`` logits.

    Args:
        depth: Number of conv blocks. Must be 2, 3, or 4.
        channels: Output channel count per block. Must have length ``depth``.
        dropout: Dropout probability in the classifier head.
        hidden_fc: Width of the hidden layer in the classifier head.
        num_classes: Output class count (default 10: 0=empty, 1-9=digits).

    Raises:
        AssertionError: if ``depth`` is outside ``{2, 3, 4}`` or
            ``len(channels) != depth``.
    """

    def __init__(
        self,
        depth: int,
        channels: List[int],
        dropout: float = 0.3,
        hidden_fc: int = 64,
        num_classes: int = 10,
    ):
        super().__init__()
        assert depth in (2, 3, 4), f"depth must be 2, 3, or 4; got {depth}"
        assert len(channels) == depth, (
            f"channels must have length {depth}; got {len(channels)}"
        )

        self.depth = depth
        self.channels = list(channels)
        self.dropout_rate = dropout
        self.hidden_fc = hidden_fc

        layers: List[nn.Module] = []
        in_ch = 1  # grayscale input
        for block_idx, out_ch in enumerate(channels):
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            if block_idx < depth - 1:
                layers.append(nn.MaxPool2d(2))
            else:
                layers.append(nn.AdaptiveAvgPool2d(1))
            in_ch = out_ch

        self.features = nn.Sequential(*layers)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[-1], hidden_fc),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_fc, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def count_parameters(model: nn.Module) -> int:
    """Total number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def verify_baseline_equivalence() -> int:
    """Return the parameter count for the baseline config and assert it
    matches the production SudokuCNN exactly. Used as a self-test."""
    baseline = SudokuCNNAblation(
        depth=3,
        channels=[32, 64, 128],
        dropout=0.3,
        hidden_fc=64,
    )
    count = count_parameters(baseline)
    expected = 102_026  # verified in app/ml/checkpoints/training_results.json
    assert count == expected, (
        f"baseline ablation model has {count} params but production SudokuCNN has {expected}; "
        "the architecture families are not equivalent"
    )
    return count


if __name__ == "__main__":
    # Self-test: baseline equivalence + a range of configurations
    from itertools import product

    print(f"baseline equivalence: {verify_baseline_equivalence():,} params ✓\n")

    depths = [2, 3, 4]
    channel_schemes = {
        "small":  {2: [16, 32],       3: [16, 32, 64],    4: [16, 32, 64, 128]},
        "medium": {2: [32, 64],       3: [32, 64, 128],   4: [32, 64, 128, 256]},
        "large":  {2: [64, 128],      3: [64, 128, 256],  4: [64, 128, 256, 512]},
    }
    dropouts = [0.2, 0.3, 0.5]

    print(f"{'depth':>5}  {'channels':<24}  {'dropout':>7}  {'params':>12}")
    print("-" * 60)
    for depth in depths:
        for cname in ["small", "medium", "large"]:
            channels = channel_schemes[cname][depth]
            for dropout in dropouts:
                model = SudokuCNNAblation(
                    depth=depth,
                    channels=channels,
                    dropout=dropout,
                    hidden_fc=64,
                )
                n = count_parameters(model)
                print(f"{depth:>5}  {str(channels):<24}  {dropout:>7.2f}  {n:>12,}")
