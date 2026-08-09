"""
Evaluate a pretrained SSL model using a trained linear probe.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from data.dataset_loader import get_dataloaders
from data.transforms import get_transforms

from models.builder_xxx import build_model
from models.classifier import LinearClassifier

from trainers.checkpoint import load_checkpoint
from trainers.evaluator import Evaluator

from utils.config import load_config
from utils.device import get_device
from utils.seed import set_seed


def parse_args():

    parser = argparse.ArgumentParser(
        description="Evaluate SSL model"
    )

    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        type=str,
    )

    parser.add_argument(
        "--ssl-checkpoint",
        required=True,
        type=str,
        help="Path to pretrained SSL checkpoint.",
    )

    parser.add_argument(
        "--linear-checkpoint",
        required=True,
        type=str,
        help="Path to trained linear probe checkpoint.",
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

    set_seed(cfg.seed)

    device = get_device()

    print(f"Using device: {device}")

    # ======================================================
    # Transforms
    # ======================================================

    _, eval_transform = get_transforms(
        method=cfg.model.name,
        dataset=cfg.dataset.name,
    )

    # ======================================================
    # Data
    # ======================================================

    dataloaders = get_dataloaders(
        dataset_name=cfg.dataset.name,
        dataset_root=cfg.dataset.root,
        train_transform=eval_transform,
        test_transform=eval_transform,
        batch_size=cfg.linear_probe.batch_size,
        val_split=cfg.dataset.val_split,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
    )

    test_loader = dataloaders["test"]

    # ======================================================
    # SSL Model
    # ======================================================

    ssl_model = build_model(cfg)

    load_checkpoint(
        args.ssl_checkpoint,
        ssl_model,
        map_location=device,
    )

    backbone = ssl_model.get_encoder()

    # ======================================================
    # Linear Classifier
    # ======================================================

    classifier = LinearClassifier(
        input_dim=ssl_model.feature_dim,
        num_classes=cfg.dataset.num_classes,
    )

    load_checkpoint(
        args.linear_checkpoint,
        classifier,
        map_location=device,
    )

    # ======================================================
    # Evaluator
    # ======================================================

    evaluator = Evaluator(
        backbone=backbone,
        classifier=classifier,
        device=device,
    )

    # ======================================================
    # Metrics
    # ======================================================

    metrics = evaluator.evaluate(
        test_loader,
    )

    print("\nEvaluation Results")

    print("-" * 40)

    for key, value in metrics.items():
        print(
            f"{key:12s}: {value:.4f}"
        )

    # ======================================================
    # Save Results
    # ======================================================

    output_dir = (
        Path(cfg.output_dir)
        / cfg.model.name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    evaluator.save_metrics(
        metrics,
        output_dir,
    )

    if cfg.evaluation.save_confusion_matrix:

        cm = evaluator.confusion_matrix(
            test_loader,
        )

        evaluator.save_confusion_matrix(
            cm,
            output_dir,
        )

    if cfg.evaluation.save_tsne:

        evaluator.plot_tsne(
            test_loader,
            output_dir,
        )

    if cfg.evaluation.save_umap:

        print(
            "UMAP plotting is not implemented."
        )

    print(
        f"\nResults saved to {output_dir}"
    )


if __name__ == "__main__":
    main()