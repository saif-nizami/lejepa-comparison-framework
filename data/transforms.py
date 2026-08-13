"""
Transformation pipelines for Self-Supervised Learning.

Supported Methods:
- SimCLR
- BYOL
- VICReg
- Barlow Twins
- LeJEPA

Supported Datasets:
- CIFAR-10
- STL-10
"""

from __future__ import annotations

from torchvision import transforms

# ============================================================
# Dataset Statistics
# ============================================================

DATASET_STATS = {
    "cifar10": {
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
        "image_size": 32,
    },
    "stl10": {
        "mean": (0.4467, 0.4398, 0.4066),
        "std": (0.2603, 0.2566, 0.2713),
        "image_size": 96,
    },
}

SUPPORTED_METHODS = {
    "simclr",
    "byol",
    "vicreg",
    "barlow_twins",
    "lejepa",
}


# ============================================================
# SSL Training Transform
# ============================================================

# def build_ssl_transform(image_size: int, mean, std):
#     """
#     Augmentation pipeline used during SSL pretraining.
#     """

#     return transforms.Compose(
#         [
#             transforms.RandomResizedCrop(
#                 image_size,
#                 scale=(0.6, 1.0),
#                 interpolation=transforms.InterpolationMode.BICUBIC,
#             ),
#             transforms.RandomHorizontalFlip(p=0.5),
#             transforms.RandomApply(
#                 [
#                     transforms.ColorJitter(
#                         brightness=0.4,
#                         contrast=0.4,
#                         saturation=0.4,
#                         hue=0.1,
#                     )
#                 ],
#                 p=0.8,
#             ),
#             transforms.RandomGrayscale(p=0.2),
#             # transforms.GaussianBlur(
#             #     kernel_size=3,
#             #     sigma=(0.1, 2.0),
#             # ),
#             transforms.ToTensor(),
#             transforms.Normalize(mean, std),
#         ]
#     )

def build_ssl_transform(dataset: str, image_size: int, mean, std):

    transforms_list = [
        transforms.RandomResizedCrop(
            image_size,
            scale=(0.6, 1.0),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply(
            [
                transforms.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                    hue=0.1,
                )
            ],
            p=0.8,
        ),
        transforms.RandomGrayscale(p=0.2),
    ]

    if dataset.lower() != "cifar10":
        transforms_list.append(
            transforms.GaussianBlur(
                kernel_size=3,
                sigma=(0.1, 2.0),
            )
        )

    transforms_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return transforms.Compose(transforms_list)


# ============================================================
# Linear Probe / Validation / Test Transform
# ============================================================

def build_eval_transform(mean, std):
    """
    Standard evaluation transform.
    """

    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


# ============================================================
# Public API
# ============================================================

def get_transforms(
    method: str,
    dataset: str,
):
    """
    Returns training and evaluation transforms.

    Parameters
    ----------
    method : str
        simclr
        byol
        vicreg
        barlow_twins
        lejepa

    dataset : str
        cifar10
        stl10
    """

    method = method.lower()
    dataset = dataset.lower()

    if method not in SUPPORTED_METHODS:
        raise ValueError(
            f"Unsupported SSL method '{method}'. "
            f"Supported methods: {sorted(SUPPORTED_METHODS)}"
        )

    if dataset not in DATASET_STATS:
        raise ValueError(
            f"Unsupported dataset '{dataset}'. "
            f"Supported datasets: {list(DATASET_STATS.keys())}"
        )

    stats = DATASET_STATS[dataset]

    # train_transform = build_ssl_transform(
    #     image_size=stats["image_size"],
    #     mean=stats["mean"],
    #     std=stats["std"],
    # )
    train_transform = build_ssl_transform(
        dataset=dataset,
        image_size=stats["image_size"],
        mean=stats["mean"],
        std=stats["std"],
    )

    eval_transform = build_eval_transform(
        mean=stats["mean"],
        std=stats["std"],
    )

    return train_transform, eval_transform