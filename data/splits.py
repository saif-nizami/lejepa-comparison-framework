"""
Dataset splitting utilities.

Provides reproducible train/validation splits
using a fixed random seed.
"""

from __future__ import annotations

import torch
from torch.utils.data import random_split


def train_val_split(
    dataset,
    val_split: float = 0.1,
    seed: int = 42,
):
    """
    Split a dataset into train and validation subsets.

    Parameters
    ----------
    dataset
        Torch dataset.

    val_split : float
        Fraction of the dataset used for validation.

    seed : int
        Random seed for reproducibility.

    Returns
    -------
    train_dataset
    val_dataset
    """

    if not 0.0 < val_split < 1.0:
        raise ValueError("val_split must be between 0 and 1.")

    generator = torch.Generator().manual_seed(seed)

    train_size = int(len(dataset) * (1 - val_split))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator,
    )

    return train_dataset, val_dataset