"""
NT-Xent Loss (SimCLR)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import normalize


class NTXentLoss(nn.Module):

    def __init__(
        self,
        temperature: float = 0.5,
    ):
        super().__init__()

        self.temperature = temperature

    def forward(self, z1, z2):

        batch_size = z1.size(0)

        z1 = normalize(z1)
        z2 = normalize(z2)

        representations = torch.cat([z1, z2], dim=0)

        similarity = torch.matmul(
            representations,
            representations.T,
        )

        similarity /= self.temperature

        mask = torch.eye(
            2 * batch_size,
            device=similarity.device,
            dtype=torch.bool,
        )

        similarity.masked_fill_(mask, float("-inf"))

        positives = torch.cat(
            [
                torch.diag(similarity, batch_size),
                torch.diag(similarity, -batch_size),
            ]
        )

        denominator = torch.logsumexp(
            similarity,
            dim=1,
        )

        loss = -positives + denominator

        return loss.mean()