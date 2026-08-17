from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler | None,
    epoch: int,
    loss: float,
) -> None:

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": (
            scheduler.state_dict()
            if scheduler is not None
            else None
        ),
    }

    torch.save(checkpoint, path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """
    Load a checkpoint.

    Returns
    -------
    dict
        Loaded checkpoint dictionary.
    """

    checkpoint = torch.load(
        path,
        map_location=map_location,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if (
        optimizer is not None
        and checkpoint.get("optimizer_state_dict") is not None
    ):
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    if (
        scheduler is not None
        and checkpoint.get("scheduler_state_dict") is not None
    ):
        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

    return checkpoint


def save_best_checkpoint(
    checkpoint_dir: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler | None,
    epoch: int,
    loss: float,
    filename="best_model.pth",
) -> None:
    """
    Save the best-performing checkpoint.
    """

    save_checkpoint(
        # Path(checkpoint_dir) / "best_model.pth",
        Path(checkpoint_dir) / filename,
        model,
        optimizer,
        scheduler,
        epoch,
        loss,
    )


def save_last_checkpoint(
    checkpoint_dir: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler | None,
    epoch: int,
    loss: float,
    filename="last_model.pth",
) -> None:
    """
    Save the latest checkpoint.
    """

    save_checkpoint(
        # Path(checkpoint_dir) / "last_model.pth",
        Path(checkpoint_dir) / filename,
        model,
        optimizer,
        scheduler,
        epoch,
        loss,
    )