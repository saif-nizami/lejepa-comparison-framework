"""
Self-Supervised Learning dataset wrapper.

Wraps any PyTorch-compatible dataset and returns
multiple independently augmented views of each image.
"""

from __future__ import annotations

from typing import Any

from torch.utils.data import Dataset


class SSLDataset(Dataset):
    """
    Dataset wrapper for Self-Supervised Learning.

    Parameters
    ----------
    dataset : Dataset
        Base dataset.

    transform
        Augmentation pipeline.

    num_views : int, default=2
        Number of augmented views to generate.

    Returns
    -------
    tuple
        (views, label)

    Example
    -------
    >>> views, label = dataset[0]

    >>> view1, view2 = views
    """

    def __init__(
        self,
        dataset: Dataset,
        transform,
        num_views: int = 2,
    ) -> None:

        if num_views < 2:
            raise ValueError(
                "num_views must be at least 2."
            )

        self.dataset = dataset
        self.transform = transform
        self.num_views = num_views

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[tuple[Any, ...], int]:

        image, label = self.dataset[index]

        views = tuple(
            self.transform(image)
            for _ in range(self.num_views)
        )

        return views, label