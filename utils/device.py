"""
Device utilities.
"""

from __future__ import annotations

import torch


def get_device() -> torch.device:
    """
    Select the best available compute device.

    Priority:
        CUDA -> MPS -> CPU
    """

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")