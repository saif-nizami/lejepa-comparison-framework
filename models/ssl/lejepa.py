"""
LeJEPA: Provable and Scalable Self-Supervised Learning
Without the Heuristics.

This implementation uses the project's SIGReg loss with the common SSL
framework used in this project.

Paper:
LeJEPA: Provable and Scalable Self-Supervised Learning
Without the Heuristics
(Balestriero & LeCun, 2025)
"""

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

    # =========================================================
    # Invariance Loss
    # =========================================================

    # @staticmethod
    # def invariance_loss(
    #     projections: torch.Tensor,
    # ) -> torch.Tensor:
    #     """
    #     Official LeJEPA invariance loss.
    #     """

    #     return (
    #         projections.mean(dim=0) - projections
    #     ).square().mean()

    # =========================================================
    # Prediction Loss
    # =========================================================

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

    # def compute_loss(
    #     self,
    #     view1: torch.Tensor,
    #     view2: torch.Tensor,
    # ) -> torch.Tensor:
    #     """
    #     Compute the LeJEPA objective.

    #     Parameters
    #     ----------
    #     view1 : torch.Tensor
    #         First augmented view.

    #     view2 : torch.Tensor
    #         Second augmented view.
    #     """

    #     z1 = self.forward(view1)

    #     z2 = self.forward(view2)

    #     print("Raw cosine:",
    #         torch.nn.functional.cosine_similarity(
    #             z1,
    #             z2,
    #             dim=1
    #         ).mean().item())

    #     print("Mean |z1-z2|:",
    #         (z1 - z2).abs().mean().item())

    #     breakpoint = False

    #     print(
    #         "z1 mean:",
    #         z1.mean().item(),
    #         "std:",
    #         z1.std().item(),
    #     )

    #     print(
    #         "z2 mean:",
    #         z2.mean().item(),
    #         "std:",
    #         z2.std().item(),
    #     )

    #     # projections = torch.stack(
    #     #     [z1, z2],
    #     #     dim=0,
    #     # )

    #     # inv_loss = self.invariance_loss(
    #     #     projections
    #     # )

    #     # sigreg_loss = self.sigreg(
    #     #     projections
    #     # )

    #     # total_loss = (
    #     #     self.lambda_sigreg * sigreg_loss
    #     #     + (1.0 - self.lambda_sigreg) * inv_loss
    #     # )

    #     projections = torch.cat(
    #         [z1, z2],
    #         dim=0,
    #     )

    #     pred_loss = self.prediction_loss(
    #         z1,
    #         z2,
    #     )

    #     # sigreg_loss = self.sigreg(
    #     #     projections,
    #     # )

    #     projections = torch.nn.functional.normalize(
    #         projections,
    #         dim=1,
    #     )

    #     sigreg_loss = self.sigreg(projections)

    #     print(
    #         f"Pred: {pred_loss.item():.4f} "
    #         f"SIGReg: {sigreg_loss.item():.4f}"
    #     )

    #     print("z1 std:", z1.std().item())
    #     print("z2 std:", z2.std().item())
    #     print(
    #         "Cos:",
    #         F.cosine_similarity(z1, z2, dim=1).mean().item()
    #     )

    #     # total_loss = (
    #     #     (1.0 - self.lambda_sigreg) * pred_loss
    #     #     + self.lambda_sigreg * sigreg_loss
    #     # )

    #     total_loss = pred_loss + 0.001 * sigreg_loss

    #     return total_loss

    def compute_loss(
        self,
        view1: torch.Tensor,
        view2: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the LeJEPA invariance and SIGReg objective."""

        z1 = self.forward(view1)
        z2 = self.forward(view2)

        # ----------------------------------------
        # Debug (print only once)
        # ----------------------------------------
        if not hasattr(self, "_debug_printed"):
            print(
                "Projector cosine:",
                F.cosine_similarity(
                    z1,
                    z2,
                    dim=1,
                ).mean().item()
            )

            self._debug_printed = True

        # ----------------------------------------
        projections = torch.stack(
            [z1, z2],
            dim=0,
        )

        invariance_loss = (
            projections - projections.mean(dim=0, keepdim=True)
        ).square().mean()

        sigreg_loss = self.sigreg(
            projections.flatten(0, 1),
        )

        total_loss = (
            (1.0 - self.lambda_sigreg) * invariance_loss
            + self.lambda_sigreg * sigreg_loss
        )

        if not hasattr(self, "_loss_debug_printed"):

            print(
                f"Invariance Loss : {invariance_loss.item():.6f}"
            )

            print(
                f"SIGReg Loss : {sigreg_loss.item():.6f}"
            )

            print(
                "Embedding std :",
                z1.std(dim=0).mean().item(),
            )

            print(
                "Embedding var :",
                z1.var(dim=0).mean().item(),
            )

            print(
                "Embedding mean:",
                z1.mean().item(),
            )

            print(
                "Embedding norm:",
                z1.norm(dim=1).mean().item(),
            )

            self._loss_debug_printed = True

        return total_loss

    # =========================================================
    # Feature Extraction
    # =========================================================

    def get_backbone(self):
        """
        Return backbone for downstream evaluation.
        """

        return self.backbone
