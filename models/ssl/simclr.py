"""
SimCLR implementation.

Paper:
A Simple Framework for Contrastive Learning of Visual Representations
(Chen et al., 2020)
"""

from __future__ import annotations
from utils.registry import register_ssl_model

import torch

from models.heads import ProjectionHead
from models.ssl.base_ssl import BaseSSLModel
from losses import NTXentLoss

@register_ssl_model("simclr")
class SimCLR(BaseSSLModel):
    """
    SimCLR model.

    Architecture:

        Image
          │
      ResNet18
          │
      Projection Head
          │
      NT-Xent Loss
    """

    def __init__(
        self,
        projection_dim: int = 2048,
        hidden_dim: int = 2048,
        temperature: float = 0.5,
        pretrained_backbone: bool = False,
    ):
        super().__init__(
            pretrained_backbone=pretrained_backbone,
        )

        self.projection_head = ProjectionHead(
            input_dim=self.feature_dim,
            hidden_dim=hidden_dim,
            output_dim=projection_dim,
        )

        self.loss_fn = NTXentLoss(
            temperature=temperature,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder and projection head.
        """

        features = self.encode(x)

        projections = self.projection_head(features)

        return projections

    def compute_loss(
        self,
        view1: torch.Tensor,
        view2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute SimCLR loss from two augmented views.
        """

        z1 = self.forward(view1)
        z2 = self.forward(view2)

        return self.loss_fn(z1, z2)

    @torch.no_grad()
    def extract_features(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract backbone representations for
        linear probing.
        """

        return self.encode(x)