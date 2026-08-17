from __future__ import annotations

import torch

from models.heads import ProjectionHead
from models.losses import SIGRegLoss
from models.ssl.base_ssl import BaseSSLModel
from utils.registry import register_ssl_model

import torch.nn.functional as F

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
        # SIGReg
        # -----------------------------------------------------

        self.sigreg = SIGRegLoss(
            knots=epps_points,
            t_max=epps_tmax,
            num_projections=num_slices,
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

    @staticmethod
    def prediction_loss(
        z1: torch.Tensor,
        z2: torch.Tensor,
    ) -> torch.Tensor:
        """
        LeJEPA prediction/alignment loss.
        """

        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # return F.mse_loss(z1, z2)
        return 2 - 2 * (
            z1 * z2
        ).sum(dim=1).mean()

    # =========================================================
    # Loss
    # =========================================================

    def compute_loss(
        self,
        view1: torch.Tensor,
        view2: torch.Tensor,
    ) -> torch.Tensor:

        z1 = self.forward(view1)
        z2 = self.forward(view2)

        invariance_loss = ((z1 - z2) ** 2).mean()

        pred_loss = self.prediction_loss(
            z1,
            z2,
        )

        projections = torch.cat(
            [z1, z2],
            dim=0,
        )

        projections = F.normalize(
            projections,
            dim=1,
        )

        sigreg_loss = self.sigreg(projections)

        total_loss = (
            pred_loss
            + 0.005 * sigreg_loss
        )

        # ----------------------------------------------------
        # Print every 10 epochs (only first batch)
        # ----------------------------------------------------
        if (
            self.current_epoch % 10 == 0
            and not self.debug_printed
        ):

            cosine = F.cosine_similarity(
                F.normalize(z1, dim=1),
                F.normalize(z2, dim=1),
                dim=1,
            ).mean()

            print()

            print(f"Epoch {self.current_epoch + 1}")

            print(
                f"Prediction Loss : {pred_loss.item():.6f}"
            )

            print(
                f"SIGReg Loss     : {sigreg_loss.item():.6f}"
            )

            print(
                f"Cosine Similarity : {cosine.item():.6f}"
            )

            print(
                f"Embedding Std : {z1.std(dim=0).mean().item():.6f}"
            )

            print(
                f"Embedding Norm : {z1.norm(dim=1).mean().item():.6f}"
            )

            self.debug_printed = True

        return total_loss

    # =========================================================
    # Feature Extraction
    # =========================================================

    def get_backbone(self):
        """
        Return backbone for downstream evaluation.
        """

        return self.backbone
