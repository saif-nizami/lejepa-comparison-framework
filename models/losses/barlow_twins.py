"""
Barlow Twins Loss
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .utils import off_diagonal


class BarlowTwinsLoss(nn.Module):

    def __init__(
        self,
        lambd: float = 5e-3,
    ):
        super().__init__()

        self.lambd = lambd

    def forward(self, z1, z2):

        batch_size = z1.size(0)

        z1 = (z1 - z1.mean(0)) / z1.std(0)
        z2 = (z2 - z2.mean(0)) / z2.std(0)

        c = torch.mm(z1.T, z2)

        c /= batch_size

        on_diag = torch.diagonal(c).add(-1).pow(2).sum()

        off_diag = off_diagonal(c).pow(2).sum()

        return on_diag + self.lambd * off_diag