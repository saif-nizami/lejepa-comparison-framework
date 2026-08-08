"""
Projection Head used by Self-Supervised Learning (SSL) methods.

Maps backbone feature representations into a latent embedding space.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    """
    Generic MLP Projection Head.

    Default architecture:
        Input → Linear → BatchNorm → ReLU → Linear

    Example:
        512 → 2048 → 2048
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 2048,
        output_dim: int = 2048,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Initialize all learnable parameters.
        """

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight,
                    nonlinearity="relu",
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @property
    def feature_dim(self) -> int:
        """
        Output feature dimension.
        """
        return self.output_dim

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through the projection head.
        """
        return self.network(x)