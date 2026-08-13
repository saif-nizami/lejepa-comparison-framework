"""
Linear probing script.

Loads a pretrained SSL backbone and trains
a linear classifier on frozen features.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from data.dataset_loader import get_dataloaders
from data.transforms import get_transforms

from models.builder_xxx import build_model
from models.classifier import LinearClassifier

from trainers.linear_probe import LinearProbeTrainer
from trainers.optimizer import build_optimizer
from trainers.scheduler import build_scheduler
from trainers.checkpoint import load_checkpoint

from utils.config import load_config
from utils.device import get_device
from utils.seed import set_seed


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/default.yaml",
    )

    parser.add_argument(
        "--model",
        default=None,
        choices=[
            "simclr",
            "byol",
            "vicreg",
            "barlow_twins",
            "lejepa",
        ],
    )

    parser.add_argument(
        "--dataset",
        default=None,
        choices=[
            "cifar10",
            "stl10",
        ],
    )

    return parser.parse_args()


def main():

    args = parse_args()

    cfg = load_config(args.config)

    if args.model is not None:
        cfg.model.name = args.model

    if args.dataset is not None:
        cfg.dataset.name = args.dataset

    set_seed(cfg.seed)

    device = get_device()

    print(f"Using device: {device}")

    _, eval_transform = get_transforms(
        cfg.model.name,
        cfg.dataset.name,
    )

    dataloaders = get_dataloaders(
        dataset_name=cfg.dataset.name,
        dataset_root=cfg.dataset.root,
        train_transform=eval_transform,
        test_transform=eval_transform,
        batch_size=cfg.linear_probe.batch_size,
        val_split=cfg.dataset.val_split,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
        ssl=False
    )

    ssl_model = build_model(cfg)

    checkpoint = (
        Path(cfg.checkpoint_dir)
        / cfg.dataset.name.lower()
        / cfg.model.name.lower()
        / "best_model.pth"
    )

    print(f"Loading {checkpoint}")

    load_checkpoint(
        checkpoint,
        ssl_model,
        map_location=device,
    )

    for name, param in ssl_model.named_parameters():
        print(name, param.abs().mean().item())
        break

    classifier = LinearClassifier(
        input_dim=ssl_model.feature_dim,
        num_classes=cfg.dataset.num_classes,
    )

    optimizer = build_optimizer(
        classifier,
        cfg,
        learning_rate=cfg.linear_probe.learning_rate,
    )

    scheduler = build_scheduler(
        optimizer,
        cfg,
    )

    trainer = LinearProbeTrainer(
        backbone=ssl_model.get_backbone(),
        classifier=classifier,
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=cfg,
    )

    best_acc = trainer.fit()

    print(f"\nBest Validation Accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()