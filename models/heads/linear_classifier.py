"""
Linear classifier used for downstream evaluation.
"""

from __future__ import annotations

import torch.nn as nn


class LinearClassifier(nn.Module):

    def __init__(
        self,
        input_dim: int = 512,
        num_classes: int = 10,
    ):
        super().__init__()

        self.classifier = nn.Linear(
            input_dim,
            num_classes,
        )

    def forward(self, x):
        return self.classifier(x)