"""
Main training script for Self-Supervised Learning.
"""

from __future__ import annotations

import argparse

from data.dataset_loader import get_dataloaders
from data.transforms import get_transforms

from models.builder_xxx import build_model

from trainers.optimizer import build_optimizer
from trainers.scheduler import build_scheduler
from trainers.trainer import Trainer

from utils.config import load_config
from utils.device import get_device
from utils.seed import set_seed


def parse_args():

    parser = argparse.ArgumentParser(
        description="SSL Training"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=[
            "simclr",
            "byol",
            "vicreg",
            "barlow_twins",
            "lejepa",
        ],
        help="Override model specified in the configuration file.",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    # ======================================================
    # Configuration
    # ======================================================

    cfg = load_config(args.config)
    
    if args.model is not None:
        cfg.model.name = args.model

    # ======================================================
    # Reproducibility
    # ======================================================

    set_seed(cfg.seed)

    # ======================================================
    # Device
    # ======================================================

    device = get_device()

    print(f"Using device: {device}")

    # ======================================================
    # Transforms
    # ======================================================

    train_transform, eval_transform = get_transforms(
        method=cfg.model.name,
        dataset=cfg.dataset.name,
    )

    # ======================================================
    # Data
    # ======================================================

    dataloaders = get_dataloaders(
        dataset_name=cfg.dataset.name,
        dataset_root=cfg.dataset.root,
        train_transform=train_transform,
        test_transform=eval_transform,
        batch_size=cfg.training.batch_size,
        val_split=cfg.dataset.val_split,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
    )

    # ======================================================
    # Model
    # ======================================================

    model = build_model(cfg)

    print(
        f"Model: {cfg.model.name}"
    )

    # ======================================================
    # Optimizer
    # ======================================================

    optimizer = build_optimizer(
        model,
        cfg,
    )

    # ======================================================
    # Scheduler
    # ======================================================

    scheduler = build_scheduler(
        optimizer,
        cfg,
    )

    # ======================================================
    # Trainer
    # ======================================================

    trainer = Trainer(
        model=model,
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=cfg,
    )

    # ======================================================
    # Train
    # ======================================================

    trainer.fit()

    print("\nTraining completed.")


if __name__ == "__main__":
    main()