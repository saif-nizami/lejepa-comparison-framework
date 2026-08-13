"""
Base class for all Self-Supervised Learning (SSL) models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from models.backbone import ResNet18Backbone


class BaseSSLModel(nn.Module, ABC):
    """
    Abstract base class for all Self-Supervised Learning methods.
    """

    def __init__(
        self,
        pretrained_backbone: bool = False,
    ) -> None:
        super().__init__()

        self.backbone = ResNet18Backbone(
            pretrained=pretrained_backbone,
        )

        self.feature_dim = self.backbone.feature_dim

    # ==========================================================
    # Encoder Interface
    # ==========================================================

    def get_encoder(self) -> nn.Module:
        """
        Return the feature extraction encoder.

        This method is used by downstream tasks such as
        linear probing and feature extraction.
        """
        return self.backbone

    def get_backbone(self) -> nn.Module:
        """
        Backward-compatible alias for get_encoder().
        """
        return self.get_encoder()

    def encode(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract encoder features.
        """
        return self.get_encoder()(x)

    @torch.no_grad()
    def extract_features(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract frozen encoder features for downstream evaluation.
        """
        self.get_encoder().eval()
        return self.encode(x)

    # ==========================================================
    # Utilities
    # ==========================================================

    def freeze_backbone(self) -> None:
        """
        Freeze encoder parameters.
        """
        for parameter in self.get_encoder().parameters():
            parameter.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """
        Unfreeze encoder parameters.
        """
        for parameter in self.get_encoder().parameters():
            parameter.requires_grad = True

    # ==========================================================
    # Optional Hook
    # ==========================================================

    def update_target_network(self) -> None:
        """
        Hook for EMA-based methods (e.g. BYOL).

        Models that do not require a target network
        can safely ignore this.
        """
        pass

    # ==========================================================
    # Abstract Interface
    # ==========================================================

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_loss(
        self,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute the self-supervised learning loss.
        """
        raise NotImplementedError
