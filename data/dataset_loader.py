"""
Dataset loading utilities.

Supported datasets:
- CIFAR-10
- STL-10

Returns reproducible PyTorch DataLoaders for
training, validation and testing.
"""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, STL10

from data.splits import train_val_split

from data.ssl_dataset import SSLDataset


SUPPORTED_DATASETS = {
    "cifar10",
    "stl10",
}


def get_dataloaders(
    dataset_name: str,
    dataset_root: str,
    train_transform,
    test_transform,
    batch_size: int = 256,
    val_split: float = 0.1,
    num_workers: int = 4,
    seed: int = 42,
):
    """
    Create train, validation and test dataloaders.

    Parameters
    ----------
    dataset_name : str
        Dataset name ("cifar10" or "stl10")

    dataset_root : str
        Dataset directory.

    train_transform
        Transform applied to training images.

    test_transform
        Transform applied to validation/test images.

    batch_size : int
        Batch size.

    val_split : float
        Validation split ratio.

    num_workers : int
        Number of DataLoader workers.

    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary containing:
        {
            "train": DataLoader,
            "val": DataLoader,
            "test": DataLoader
        }
    """

    dataset_name = dataset_name.lower()

    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. "
            f"Supported datasets: {sorted(SUPPORTED_DATASETS)}"
        )

    dataset_root = Path(dataset_root)

    # ==========================================================
    # CIFAR-10
    # ==========================================================

    if dataset_name == "cifar10":

        train_dataset = CIFAR10(
            root=dataset_root,
            train=True,
            transform=train_transform,
            download=False,
        )

        test_dataset = CIFAR10(
            root=dataset_root,
            train=False,
            transform=test_transform,
            download=False,
        )

    # ==========================================================
    # STL-10
    # ==========================================================

    else:

        train_dataset = STL10(
            root=dataset_root,
            split="train",
            transform=train_transform,
            download=False,
        )

        test_dataset = STL10(
            root=dataset_root,
            split="test",
            transform=test_transform,
            download=False,
        )

    # ==========================================================
    # Train / Validation Split
    # ==========================================================

    train_dataset, val_dataset = train_val_split(
        dataset=train_dataset,
        val_split=val_split,
        seed=seed,
    )
    train_dataset = SSLDataset(
        dataset=train_dataset,
        transform=train_transform,
        num_views=2
    )

    # ==========================================================
    # DataLoaders
    # ==========================================================

    common_args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": False,  # Apple Silicon (MPS)
        "persistent_workers": num_workers > 0,
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **common_args,
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **common_args,
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **common_args,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }