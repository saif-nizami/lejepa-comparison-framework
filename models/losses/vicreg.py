"""
VICReg Loss
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .utils import off_diagonal


class VICRegLoss(nn.Module):

    def __init__(
        self,
        sim_coeff=25.0,
        std_coeff=25.0,
        cov_coeff=1.0,
    ):
        super().__init__()

        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff

    def forward(self, x, y):

        repr_loss = ((x - y) ** 2).mean()

        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 1e-4)
        std_y = torch.sqrt(y.var(dim=0) + 1e-4)

        std_loss = (
            torch.relu(1 - std_x).mean()
            + torch.relu(1 - std_y).mean()
        )

        # cov_x = (x.T @ x) / (x.size(0) - 1)
        # cov_y = (y.T @ y) / (y.size(0) - 1)

        cov_x = (x.T @ x) / (x.size(0) - 1)
        cov_y = (y.T @ y) / (y.size(0) - 1)

        # print("cov_x max :", cov_x.abs().max().item())
        # print("cov_x mean:", cov_x.abs().mean().item())

        # print("cov_x diag max :", cov_x.diag().max().item())
        # print("cov_x off max  :", off_diagonal(cov_x).abs().max().item())

        # print("cov_y diag max :", cov_y.diag().max().item())
        # print("cov_y off max  :", off_diagonal(cov_y).abs().max().item())

        cov_loss = (
            off_diagonal(cov_x).pow(2).sum() / x.size(1)
            + off_diagonal(cov_y).pow(2).sum() / y.size(1)
        )

        # print("repr :", repr_loss.item())
        # print("std  :", std_loss.item())
        # print("cov  :", cov_loss.item())

        assert torch.isfinite(repr_loss)
        assert torch.isfinite(std_loss)
        assert torch.isfinite(cov_loss)

        return (
            self.sim_coeff * repr_loss
            + self.std_coeff * std_loss
            + self.cov_coeff * cov_loss
        )

        return (
            self.sim_coeff * repr_loss
            + self.std_coeff * std_loss
            + self.cov_coeff * cov_loss
        )