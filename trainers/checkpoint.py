"""
Checkpoint utilities.
"""

from __future__ import annotations

from pathlib import Path
import torch


def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    loss: float,
    path: str | Path,
):
    """
    Save training checkpoint.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "loss": loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def load_checkpoint(
    model,
    optimizer,
    path: str | Path,
):
    """
    Load training checkpoint.
    """

    checkpoint = torch.load(
        path,
        map_location="cpu",
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    return (
        model,
        optimizer,
        checkpoint["epoch"],
        checkpoint["loss"],
    )