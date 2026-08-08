"""
Variance-Invariance-Covariance Regularization (VICReg)

Paper:
Variance-Invariance-Covariance Regularization
(Bardes et al., 2022)
"""

from __future__ import annotations

import torch

from losses import VICRegLoss
from models.heads import ProjectionHead
from models.ssl.base_ssl import BaseSSLModel
from utils.registry import register_ssl_model


@register_ssl_model("vicreg")
class VICReg(BaseSSLModel):
    """
    VICReg implementation.

    Architecture
    ------------
    Backbone
        ↓
    Projection Head
        ↓
    VICReg Loss
    """

    def __init__(
        self,
        projection_dim: int = 2048,
        hidden_dim: int = 2048,
        sim_coeff: float = 25.0,
        std_coeff: float = 25.0,
        cov_coeff: float = 1.0,
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

        self.loss_fn = VICRegLoss(
            sim_coeff=sim_coeff,
            std_coeff=std_coeff,
            cov_coeff=cov_coeff,
        )

    # ==========================================================
    # Forward
    # ==========================================================

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

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

        z1 = self.forward(view1)

        z2 = self.forward(view2)

        return self.loss_fn(z1, z2)