"""
Learning rate scheduler factory.
"""

from __future__ import annotations

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    MultiStepLR,
    OneCycleLR,
    ReduceLROnPlateau,
    StepLR,
)


def build_scheduler(
    optimizer: Optimizer,
    config,
    steps_per_epoch: int | None = None,
):
    """
    Build a learning rate scheduler from the configuration.

    Parameters
    ----------
    optimizer : Optimizer
        Optimizer instance.

    config
        Loaded configuration.

    steps_per_epoch : int, optional
        Required by OneCycleLR.

    Returns
    -------
    torch.optim.lr_scheduler.LRScheduler | None
    """

    name = config.scheduler.name.lower()

    if name == "none":
        return None

    if name == "cosineannealinglr":

        return CosineAnnealingLR(
            optimizer,
            T_max=config.scheduler.T_max,
            eta_min=config.scheduler.eta_min,
        )

    if name == "cosineannealingwarmrestarts":

        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.scheduler.T_0,
            T_mult=config.scheduler.T_mult,
            eta_min=config.scheduler.eta_min,
        )

    if name == "steplr":

        return StepLR(
            optimizer,
            step_size=config.scheduler.step_size,
            gamma=config.scheduler.gamma,
        )

    if name == "multisteplr":

        return MultiStepLR(
            optimizer,
            milestones=config.scheduler.milestones,
            gamma=config.scheduler.gamma,
        )

    if name == "reducelronplateau":

        return ReduceLROnPlateau(
            optimizer,
            mode=config.scheduler.mode,
            factor=config.scheduler.factor,
            patience=config.scheduler.patience,
            min_lr=config.scheduler.min_lr,
        )

    if name == "onecyclelr":

        if steps_per_epoch is None:
            raise ValueError(
                "steps_per_epoch is required for OneCycleLR."
            )

        return OneCycleLR(
            optimizer,
            max_lr=config.optimizer.learning_rate,
            epochs=config.training.epochs,
            steps_per_epoch=steps_per_epoch,
        )

    raise ValueError(
        f"Unsupported scheduler: {config.scheduler.name}"
    )