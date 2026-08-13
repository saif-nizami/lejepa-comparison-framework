"""
Linear probing trainer.

Trains a linear classifier on top of a frozen
pretrained backbone.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from trainers.checkpoint import save_best_checkpoint
from trainers.metrics import MetricTracker


class LinearProbeTrainer:
    """
    Trainer for linear evaluation.
    """

    def __init__(
        self,
        backbone: nn.Module,
        classifier: nn.Module,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device,
        config,
    ) -> None:

        self.device = device
        self.config = config

        self.checkpoint_dir = (
            Path(config.checkpoint_dir)
            / "linear_probe"
            / config.model.name.lower()
        )
        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # --------------------------------------------------
        # Frozen backbone
        # --------------------------------------------------

        self.backbone = backbone.to(device)

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.backbone.eval()

        # --------------------------------------------------
        # Trainable classifier
        # --------------------------------------------------

        self.classifier = classifier.to(device)

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.epochs = config.linear_probe.epochs

        self.use_amp = (
            device.type == "cuda"
            and config.training.mixed_precision
        )

        self.scaler = GradScaler(
            "cuda",
            enabled=self.use_amp,
        )

        self.criterion = nn.CrossEntropyLoss()

        self.train_metrics = MetricTracker(
            "loss",
            "accuracy",
        )

        self.val_metrics = MetricTracker(
            "loss",
            "accuracy",
        )

    # ======================================================
    # Training
    # ======================================================

    def training_step(
        self,
        batch,
    ) -> tuple[float, float]:

        images, labels = batch

        # print(images.shape)
        # print(labels.min(), labels.max())

        images = images.to(
            self.device,
            non_blocking=True,
        )

        labels = labels.to(
            self.device,
            non_blocking=True,
        )

        self.optimizer.zero_grad(
            set_to_none=True,
        )

        self.backbone.eval()

        with torch.no_grad():
            features = self.backbone(images)

        # print("Feature mean:", features.mean().item())
        # print("Feature std :", features.std().item())

        with autocast(
            device_type="cuda",
            enabled=self.use_amp,
        ):

            logits = self.classifier(features)

            loss = self.criterion(
                logits,
                labels,
            )

            # print("Raw CE Loss:", loss.item())
            # raise SystemExit

        self.scaler.scale(loss).backward()

        if self.config.training.gradient_clip is not None:

            self.scaler.unscale_(self.optimizer)

            torch.nn.utils.clip_grad_norm_(
                self.classifier.parameters(),
                self.config.training.gradient_clip,
            )

        self.scaler.step(self.optimizer)
        self.scaler.update()

        predictions = logits.argmax(dim=1)

        correct = (predictions == labels).sum().item()

        accuracy = (
            correct / labels.size(0)
        ) * 100.0

        return (
            loss.item(),
            accuracy,
        )

    # ======================================================
    # Validation
    # ======================================================

    @torch.no_grad()
    def validation_step(
        self,
        batch,
    ) -> tuple[float, float]:

        images, labels = batch

        images = images.to(
            self.device,
            non_blocking=True,
        )

        labels = labels.to(
            self.device,
            non_blocking=True,
        )

        self.backbone.eval()

        with torch.no_grad():
            features = self.backbone(images)

        with autocast(
            device_type="cuda",
            enabled=self.use_amp,
        ):

            logits = self.classifier(features)

            loss = self.criterion(
                logits,
                labels,
            )

        predictions = logits.argmax(dim=1)

        correct = (predictions == labels).sum().item()

        accuracy = (
            correct / labels.size(0)
        ) * 100.0

        return (
            loss.item(),
            accuracy,
        )

    # ======================================================
    # Train Epoch
    # ======================================================

    def train_one_epoch(
        self,
    ) -> dict[str, float]:

        self.classifier.train()

        self.train_metrics.reset()

        progress = tqdm(
            self.train_loader,
            desc="Training",
            leave=False,
            dynamic_ncols=True,
        )

        for batch in progress:

            loss, acc = self.training_step(batch)

            self.train_metrics.update(
                "loss",
                loss,
            )

            self.train_metrics.update(
                "accuracy",
                acc,
            )

            progress.set_postfix(
                loss=f"{loss:.4f}",
                acc=f"{acc:.2f}",
            )

        return self.train_metrics.as_dict()

    # ======================================================
    # Validation Epoch
    # ======================================================

    @torch.no_grad()
    def validate(
        self,
    ) -> dict[str, float]:

        self.classifier.eval()

        self.val_metrics.reset()

        progress = tqdm(
            self.val_loader,
            desc="Validation",
            leave=False,
            dynamic_ncols=True,
        )

        for batch in progress:

            loss, acc = self.validation_step(batch)

            self.val_metrics.update(
                "loss",
                loss,
            )

            self.val_metrics.update(
                "accuracy",
                acc,
            )

            progress.set_postfix(
                loss=f"{loss:.4f}",
                acc=f"{acc:.2f}",
            )

        return self.val_metrics.as_dict()

    # ======================================================
    # Fit
    # ======================================================

    def fit(
        self,
    ) -> float:

        best_acc = 0.0

        # print(sum(p.requires_grad for p in self.backbone.parameters())) 
        # print(sum(p.requires_grad for p in self.classifier.parameters()))

        # checkpoint_dir = (
        #     Path(self.config.checkpoint_dir)
        #     / "linear_probe"
        # )

        for epoch in range(self.epochs):

            train = self.train_one_epoch()

            val = self.validate()

            if self.scheduler is not None:

                if isinstance(
                    self.scheduler,
                    ReduceLROnPlateau,
                ):
                    self.scheduler.step(val["loss"])
                else:
                    self.scheduler.step()

            if val["accuracy"] > best_acc:

                best_acc = val["accuracy"]

                save_best_checkpoint(
                    self.checkpoint_dir,
                    self.classifier,
                    self.optimizer,
                    self.scheduler,
                    epoch + 1,
                    val["loss"],
                    filename="linear_probe_best.pth",
                )

            print(
                f"Epoch {epoch + 1:03d} | "
                f"Train Loss {train['loss']:.4f} | "
                f"Train Acc {train['accuracy']:.2f}% | "
                f"Val Loss {val['loss']:.4f} | "
                f"Val Acc {val['accuracy']:.2f}%"
            )

        return best_acc