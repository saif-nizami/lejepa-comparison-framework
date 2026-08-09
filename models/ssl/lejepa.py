"""
LeJEPA: Provable and Scalable Self-Supervised Learning
Without the Heuristics.

This implementation integrates the official LeJEPA package
with the common SSL framework used in this project.

Paper:
LeJEPA: Provable and Scalable Self-Supervised Learning
Without the Heuristics
(Balestriero & LeCun, 2025)
"""

from __future__ import annotations

import torch

from lejepa.multivariate import SlicingUnivariateTest
from lejepa.univariate import EppsPulley

from models.heads import ProjectionHead
from models.ssl.base_ssl import BaseSSLModel
from utils.registry import register_ssl_model


@register_ssl_model("lejepa")
class LeJEPA(BaseSSLModel):
    """
    LeJEPA implementation.

    Architecture
    ------------

        Image
           │
           ▼
      ResNet18 Backbone
           │
           ▼
      Projection Head
           │
           ▼
      Invariance Loss
             +
      Official SIGReg
    """

    def __init__(
        self,
        projection_dim: int = 128,
        hidden_dim: int = 2048,
        lambda_sigreg: float = 0.02,
        num_slices: int = 256,
        epps_points: int = 17,
        epps_tmax: float = 3.0,
        pretrained_backbone: bool = False,
    ) -> None:

        super().__init__(
            pretrained_backbone=pretrained_backbone,
        )

        self.lambda_sigreg = lambda_sigreg

        # -----------------------------------------------------
        # Projection Head
        # -----------------------------------------------------

        self.projection_head = ProjectionHead(
            input_dim=self.feature_dim,
            hidden_dim=hidden_dim,
            output_dim=projection_dim,
        )

        # -----------------------------------------------------
        # Official SIGReg
        # -----------------------------------------------------

        self.sigreg = SlicingUnivariateTest(
            univariate_test=EppsPulley(
                t_max=epps_tmax,
                n_points=epps_points,
            ),
            num_slices=num_slices,
            reduction="mean",
        )

    # =========================================================
    # Forward
    # =========================================================

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Returns projected representations.
        """

        features = self.encode(x)

        projections = self.projection_head(features)

        return projections

    # =========================================================
    # Invariance Loss
    # =========================================================

    @staticmethod
    def invariance_loss(
        projections: torch.Tensor,
    ) -> torch.Tensor:
        """
        Official LeJEPA invariance loss.
        """

        return (
            projections.mean(dim=0) - projections
        ).square().mean()

    # =========================================================
    # Loss
    # =========================================================

    def compute_loss(
        self,
        view1: torch.Tensor,
        view2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the LeJEPA objective.

        Parameters
        ----------
        view1 : torch.Tensor
            First augmented view.

        view2 : torch.Tensor
            Second augmented view.
        """

        z1 = self.forward(view1)

        z2 = self.forward(view2)

        projections = torch.stack(
            [z1, z2],
            dim=0,
        )

        inv_loss = self.invariance_loss(
            projections
        )

        sigreg_loss = self.sigreg(
            projections
        )

        total_loss = (
            self.lambda_sigreg * sigreg_loss
            + (1.0 - self.lambda_sigreg) * inv_loss
        )

        return total_loss

    # =========================================================
    # Feature Extraction
    # =========================================================

    def get_backbone(self):
        """
        Return backbone for downstream evaluation.
        """

        return self.backbone