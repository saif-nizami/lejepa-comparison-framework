# """
# Base class for all Self-Supervised Learning (SSL) models.
# """

# from __future__ import annotations

# from abc import ABC, abstractmethod

# import torch
# import torch.nn as nn

# from models.backbone import ResNet18Backbone


# class BaseSSLModel(nn.Module, ABC):
#     """
#     Base class for all SSL methods.

#     Every SSL algorithm shares:

#     - ResNet-18 backbone
#     - Feature extraction
#     - Freeze / unfreeze utilities
#     """

#     def __init__(
#         self,
#         pretrained_backbone: bool = False,
#     ):
#         super().__init__()

#         self.backbone = ResNet18Backbone(
#             pretrained=pretrained_backbone
#         )

#         self.feature_dim = self.backbone.feature_dim

#     # ---------------------------------------------------------
#     # Feature Extraction
#     # ---------------------------------------------------------

#     def encode(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Extract backbone features.
#         """
#         return self.backbone(x)

#     @torch.no_grad()
#     def extract_features(
#         self,
#         x: torch.Tensor,
#     ) -> torch.Tensor:
#         """
#         Used during linear probing.
#         """
#         return self.encode(x)

#     # ---------------------------------------------------------
#     # Backbone Utilities
#     # ---------------------------------------------------------

#     def freeze_backbone(self):

#         for param in self.backbone.parameters():
#             param.requires_grad = False

#     def unfreeze_backbone(self):

#         for param in self.backbone.parameters():
#             param.requires_grad = True

#     # ---------------------------------------------------------
#     # Optional Hooks
#     # ---------------------------------------------------------

#     def update_target_network(self):
#         """
#         Only required for teacher/student methods
#         such as BYOL and LeJEPA.
#         """
#         pass

#     # ---------------------------------------------------------
#     # Abstract Methods
#     # ---------------------------------------------------------

#     @abstractmethod
#     def compute_loss(
#         self,
#         *args,
#         **kwargs,
#     ):
#         """
#         Compute SSL loss.
#         """
#         pass

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
    Base class for all SSL methods.
    """

    def __init__(
        self,
        pretrained_backbone: bool = False,
    ) -> None:
        super().__init__()

        self.backbone = ResNet18Backbone(
            pretrained=pretrained_backbone
        )

        self.feature_dim = self.backbone.feature_dim

    # ==========================================================
    # Backbone Interface
    # ==========================================================

    def get_backbone(self) -> nn.Module:
        """
        Return the backbone used for feature extraction.

        Child classes may override this method.
        """
        return self.backbone

    def encode(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode an image into a feature vector.
        """
        return self.get_backbone()(x)

    @torch.no_grad()
    def extract_features(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract backbone features for downstream evaluation.
        """
        return self.encode(x)

    # ==========================================================
    # Utilities
    # ==========================================================

    def freeze_backbone(self) -> None:

        for parameter in self.get_backbone().parameters():
            parameter.requires_grad = False

    def unfreeze_backbone(self) -> None:

        for parameter in self.get_backbone().parameters():
            parameter.requires_grad = True

    # ==========================================================
    # Optional Hook
    # ==========================================================

    def update_target_network(self) -> None:
        """
        Override for EMA-based methods.
        """
        pass

    # ==========================================================
    # Abstract Methods
    # ==========================================================

    @abstractmethod
    def compute_loss(
        self,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute the SSL loss.
        """
        raise NotImplementedError