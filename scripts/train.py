"""
Main training script for Self-Supervised Learning.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make direct execution (`python scripts/train.py`) work as well as module
# execution (`python -m scripts.train`).
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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

    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=[
            "cifar10",
            "stl10",
        ],
        help="Override dataset specified in the configuration.",
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

    if args.dataset is not None:
        cfg.dataset.name = args.dataset

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
        ssl=True,
        ssl_method=cfg.model.name,
    )

    # ======================================================
    # Model
    # ======================================================

    model = build_model(cfg)

    print(
        f"Model: {cfg.model.name}"
    )

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    print("Model parameters:", total)

    # ======================================================
    # Optimizer
    # ======================================================

    optimizer = build_optimizer(
        model,
        cfg,
    )

    num = sum(
        p.numel()
        for group in optimizer.param_groups
        for p in group["params"]
    )

    print("Optimizer parameters:", num)

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
        # val_loader=dataloaders["val"],
        val_loader=None,
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
