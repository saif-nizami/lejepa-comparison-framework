"""
Supported methods:
- SimCLR
- BYOL
- VICReg
- Barlow Twins
- LeJEPA
"""

from __future__ import annotations

from typing import Callable, Tuple


class MultiViewTransform:
    """
    Generates multiple augmented views from the same image.

    Example
    -------
    image
        ↓
    Transform
      ↓    ↓
    View1 View2
    """

    def __init__(
        self,
        transform: Callable,
        num_views: int = 2,
    ):
        self.transform = transform
        self.num_views = num_views

    def __call__(self, image):

        return [self.transform(image) for _ in range(self.num_views)]


class LeJEPATransform:
    """
    Generates two independent views for LeJEPA.

    Future versions can use different
    context and target augmentations if required.
    """

    def __init__(
        self,
        context_transform: Callable,
        target_transform: Callable,
    ):
        self.context_transform = context_transform
        self.target_transform = target_transform

    def __call__(self, image) -> Tuple:

        context = self.context_transform(image)
        target = self.target_transform(image)

        return context, target


def get_ssl_augmentation(
    method: str,
    base_transform: Callable,
):
    """
    Returns the appropriate augmentation
    wrapper for the selected SSL method.
    """

    method = method.lower()

    if method in {
        "simclr",
        "byol",
        "vicreg",
        "barlow_twins",
    }:
        return MultiViewTransform(
            transform=base_transform,
            num_views=2,
        )

    elif method == "lejepa":
        return LeJEPATransform(
            context_transform=base_transform,
            target_transform=base_transform,
        )

    raise ValueError(
        f"Unsupported SSL method: {method}"
    )