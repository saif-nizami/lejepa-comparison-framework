"""
Shared ResNet-18 backbone for all SSL methods.

The classification layer is removed so that the network
outputs feature embeddings instead of class predictions.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import resnet18


class ResNet18Backbone(nn.Module):
    """
    ResNet-18 feature extractor.

    Output:
        (batch_size, 512)
    """

    def __init__(
        self,
        pretrained: bool = False,
    ):
        super().__init__()

        model = resnet18(weights=None if not pretrained else "DEFAULT")

        # Remove classification layer
        self.encoder = nn.Sequential(
            *list(model.children())[:-1]
        )

        self.feature_dim = model.fc.in_features

    def forward(self, x):

        x = self.encoder(x)

        x = x.flatten(start_dim=1)

        return x
