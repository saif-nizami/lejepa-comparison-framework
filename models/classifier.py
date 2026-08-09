"""
Linear classifier used for downstream evaluation.

A single fully-connected layer trained on top of a
frozen SSL encoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LinearClassifier(nn.Module):
    """
    Linear classifier for SSL evaluation.

    Parameters
    ----------
    input_dim : int
        Encoder feature dimension.

    num_classes : int
        Number of output classes.
    """

    def __init__(
        self,
        input_dim: int = 512,
        num_classes: int = 10,
    ) -> None:
        super().__init__()

        self.classifier = nn.Linear(
            input_dim,
            num_classes,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Feature vectors from the encoder.

        Returns
        -------
        torch.Tensor
            Classification logits.
        """
        return self.classifier(x)