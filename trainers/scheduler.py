"""
Learning-rate schedulers.
"""

from __future__ import annotations

import torch


def cosine_scheduler(
    optimizer,
    epochs: int,
):

    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
    )