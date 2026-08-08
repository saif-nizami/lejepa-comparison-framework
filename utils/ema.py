"""
Exponential Moving Average (EMA) utilities.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def update_moving_average(
    online_model,
    target_model,
    momentum: float = 0.996,
):
    """
    Update target network parameters using EMA.
    """

    for online_param, target_param in zip(
        online_model.parameters(),
        target_model.parameters(),
    ):
        target_param.data.mul_(momentum).add_(
            online_param.data,
            alpha=1.0 - momentum,
        )