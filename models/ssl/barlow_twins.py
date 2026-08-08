"""
Barlow Twins

Paper:
Barlow Twins: Self-Supervised Learning via Redundancy Reduction
(Zbontar et al., 2021)
"""

from __future__ import annotations

import torch

from losses import BarlowTwinsLoss
from models.heads import ProjectionHead
from models.ssl.base_ssl import BaseSSLModel
from utils.registry import register_ssl_model


@register_ssl_model("barlow_twins")
class BarlowTwins(BaseSSLModel):
    """
    Barlow Twins implementation.

    Architecture
    ------------
    Backbone
        ↓
    Projection Head
        ↓
    Barlow Twins Loss
    """

    def __init__(
        self,
        projection_dim: int = 2048,
        hidden_dim: int = 2048,
        lambd: float = 5e-3,
        pretrained_backbone: bool = False,
    ) -> None:

        super().__init__(
            pretrained_backbone=pretrained_backbone,
        )

        self.projection_head = ProjectionHead(
            input_dim=self.feature_dim,
            hidden_dim=hidden_dim,
            output_dim=projection_dim,
        )

        self.loss_fn = BarlowTwinsLoss(
            lambd=lambd,
        )

    # ==========================================================
    # Forward
    # ==========================================================

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through the backbone and projection head.
        """

        features = self.encode(x)

        projections = self.projection_head(features)

        return projections

    # ==========================================================
    # Loss
    # ==========================================================

    def compute_loss(
        self,
        view1: torch.Tensor,
        view2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the Barlow Twins loss.
        """

        z1 = self.forward(view1)

        z2 = self.forward(view2)

        return self.loss_fn(z1, z2)