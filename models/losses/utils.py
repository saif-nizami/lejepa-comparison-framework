"""
Shared mathematical utilities for SSL losses.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def normalize(z: torch.Tensor) -> torch.Tensor:
    """
    L2-normalize feature vectors.
    """
    return F.normalize(z, dim=1)


def cosine_similarity(
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """
    Cosine similarity between two batches.
    """
    x = normalize(x)
    y = normalize(y)

    return (x * y).sum(dim=1)


def off_diagonal(matrix: torch.Tensor) -> torch.Tensor:
    """
    Return off-diagonal elements of a square matrix.
    """
    n, m = matrix.shape

    if n != m:
        raise ValueError("Input matrix must be square.")

    return matrix.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()