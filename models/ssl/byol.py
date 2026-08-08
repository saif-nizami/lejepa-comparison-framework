"""
Bootstrap Your Own Latent (BYOL)

Paper:
Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning
(Grill et al., 2020)
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from losses import NegativeCosineSimilarity
from models.heads import PredictorHead, ProjectionHead
from models.ssl.base_ssl import BaseSSLModel
from utils.ema import update_moving_average
from utils.registry import register_ssl_model


@register_ssl_model("byol")
class BYOL(BaseSSLModel):
    """
    Bootstrap Your Own Latent (BYOL)

    Architecture
    ------------
    Online Network
        Backbone
            ↓
        Projection Head
            ↓
        Predictor Head

    Target Network
        Backbone
            ↓
        Projection Head

    The target network is updated using an
    Exponential Moving Average (EMA).
    """

    def __init__(
        self,
        projection_dim: int = 2048,
        hidden_dim: int = 2048,
        predictor_hidden_dim: int = 512,
        momentum: float = 0.996,
        pretrained_backbone: bool = False,
    ) -> None:

        super().__init__(
            pretrained_backbone=pretrained_backbone,
        )

        self.momentum = momentum

        # -----------------------------------------------------
        # Online Network
        # -----------------------------------------------------

        self.online_backbone = self.backbone

        self.online_projection = ProjectionHead(
            input_dim=self.feature_dim,
            hidden_dim=hidden_dim,
            output_dim=projection_dim,
        )

        self.online_predictor = PredictorHead(
            input_dim=projection_dim,
            hidden_dim=predictor_hidden_dim,
            output_dim=projection_dim,
        )

        # -----------------------------------------------------
        # Target Network
        # -----------------------------------------------------

        self.target_backbone = copy.deepcopy(
            self.online_backbone
        )

        self.target_projection = copy.deepcopy(
            self.online_projection
        )

        self._freeze_target_network()

        # -----------------------------------------------------
        # Loss
        # -----------------------------------------------------

        self.loss_fn = NegativeCosineSimilarity()

    # =========================================================
    # Target Network
    # =========================================================

    def _freeze_target_network(self) -> None:
        """
        Disable gradients for the target network.
        """

        for parameter in self.target_backbone.parameters():
            parameter.requires_grad = False

        for parameter in self.target_projection.parameters():
            parameter.requires_grad = False

    # =========================================================
    # Online Forward
    # =========================================================

    def _forward_online(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through the online network.
        """

        features = self.online_backbone(x)

        projections = self.online_projection(features)

        predictions = self.online_predictor(projections)

        return predictions

    # =========================================================
    # Target Forward
    # =========================================================

    @torch.no_grad()
    def _forward_target(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through the target network.
        """

        features = self.target_backbone(x)

        projections = self.target_projection(features)

        return projections

    # =========================================================
    # Standard Forward
    # =========================================================

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Default forward pass.

        Returns predictions from the online network.
        """

        return self._forward_online(x)

        # =========================================================
    # BYOL Loss
    # =========================================================

    def compute_loss(
        self,
        view1: torch.Tensor,
        view2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the symmetric BYOL loss.

        Parameters
        ----------
        view1 : torch.Tensor
            First augmented view.

        view2 : torch.Tensor
            Second augmented view.

        Returns
        -------
        torch.Tensor
            Scalar BYOL loss.
        """

        # -----------------------------
        # Online Network
        # -----------------------------

        p1 = self._forward_online(view1)
        p2 = self._forward_online(view2)

        # -----------------------------
        # Target Network
        # -----------------------------

        with torch.no_grad():
            z1 = self._forward_target(view1)
            z2 = self._forward_target(view2)

        # -----------------------------
        # Symmetric BYOL Loss
        # -----------------------------

        loss = (
            self.loss_fn(p1, z2)
            + self.loss_fn(p2, z1)
        ) * 0.5

        return loss

    # =========================================================
    # EMA Update
    # =========================================================

    @torch.no_grad()
    def update_target_network(self) -> None:
        """
        Update the target network using an
        Exponential Moving Average (EMA).
        """

        update_moving_average(
            self.online_backbone,
            self.target_backbone,
            momentum=self.momentum,
        )

        update_moving_average(
            self.online_projection,
            self.target_projection,
            momentum=self.momentum,
        )

    # =========================================================
    # Feature Extraction
    # =========================================================

    # @torch.no_grad()
    # def extract_features(
    #     self,
    #     x: torch.Tensor,
    # ) -> torch.Tensor:
    #     """
    #     Extract backbone features for
    #     downstream linear evaluation.
    #     """

    #     return self.online_backbone(x)

    def get_backbone(self) -> nn.Module:
        """
        Return the online backbone for downstream evaluation.
        """
        return self.online_backbone