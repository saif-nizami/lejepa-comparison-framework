# """
# SIGReg regularization.

# Placeholder implementation.

# The complete implementation will be added when
# integrating the official LeJEPA repository.
# """

# from __future__ import annotations

# import torch.nn as nn


# class SIGRegLoss(nn.Module):

#     def __init__(self):
#         super().__init__()

#     def forward(self, loss):

#         return loss

"""
SIGReg loss used by LeJEPA.

Official implementation adapted from the LeJEPA
minimal example.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SIGRegLoss(nn.Module):
    """
    Spectral Integral Gaussian Regularization.
    """

    def __init__(
        self,
        knots: int = 17,
        t_max: float = 3.0,
        num_projections: int = 256,
    ) -> None:

        super().__init__()

        self.num_projections = num_projections

        t = torch.linspace(
            0,
            t_max,
            knots,
            dtype=torch.float32,
        )

        dt = t_max / (knots - 1)

        weights = torch.full(
            (knots,),
            2 * dt,
            dtype=torch.float32,
        )

        weights[[0, -1]] = dt

        window = torch.exp(
            -(t ** 2) / 2
        )

        self.register_buffer("t", t)

        self.register_buffer("phi", window)

        self.register_buffer(
            "weights",
            weights * window,
        )

    def forward(
        self,
        projections: torch.Tensor,
    ) -> torch.Tensor:

        feature_dim = projections.size(-1)

        A = torch.randn(
            feature_dim,
            self.num_projections,
            device=projections.device,
        )

        A = A / A.norm(
            p=2,
            dim=0,
            keepdim=True,
        )

        x_t = (
            projections @ A
        ).unsqueeze(-1) * self.t

        err = (
            (x_t.cos().mean(-2) - self.phi).square()
            + x_t.sin().mean(-2).square()
        )

        statistic = (
            err @ self.weights
        ) * projections.size(0)

        return statistic.mean()