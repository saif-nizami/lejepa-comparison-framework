from __future__ import annotations

from typing import Any

from torch.utils.data import Dataset


class SSLDataset(Dataset):

    def __init__(
        self,
        dataset: Dataset,
        transform,
        num_views: int = 2,
        transform_returns_views: bool = False,
    ) -> None:

        if num_views < 2:
            raise ValueError(
                "num_views must be at least 2."
            )

        self.dataset = dataset
        self.transform = transform
        self.num_views = num_views
        self.transform_returns_views = transform_returns_views

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[tuple[Any, ...], int]:

        # import torch

        # image, label = self.dataset[index]

        # view1 = self.transform(image)
        # view2 = self.transform(image)

        # print("Equal:", torch.equal(view1, view2))

        # return (view1, view2), label

        image, label = self.dataset[index]

        if self.transform_returns_views:
            views = tuple(self.transform(image))

            if len(views) != self.num_views:
                raise ValueError(
                    "Multi-view transform returned an unexpected number of views."
                )
        else:
            views = tuple(
                self.transform(image)
                for _ in range(self.num_views)
            )

        return views, label
