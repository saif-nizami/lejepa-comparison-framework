"""
Generic trainer for Self-Supervised Learning models.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.amp import GradScaler
from tqdm import tqdm

from models.ssl import BaseSSLModel

from trainers.checkpoint import (
    save_best_checkpoint,
    save_last_checkpoint,
)

from trainers.logger import TrainingLogger
from trainers.metrics import MetricTracker


class Trainer:
    """
    Generic SSL Trainer.

    Supports:
        - SimCLR
        - BYOL
        - VICReg
        - Barlow Twins
        - LeJEPA
    """

    def __init__(
        self,
        model: BaseSSLModel,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device,
        config,
    ) -> None:

        self.model = model.to(device)

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.device = device
        self.config = config

        self.epochs = config.training.epochs

        self.current_epoch = 0
        self.best_loss = float("inf")

        # Enable AMP only on CUDA
        self.use_amp = (
            self.device.type == "cuda"
            and self.config.training.mixed_precision
        )

        self.scaler = GradScaler(
            "cuda",
            enabled=self.use_amp,
        )

        # Checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Logger
        self.logger = TrainingLogger(
            log_dir=config.log_dir,
            experiment_name=config.experiment_name,
            use_tensorboard=config.logging.tensorboard,
            use_csv=config.logging.csv,
        )

        # Metrics
        self.train_metrics = MetricTracker("loss")
        self.val_metrics = MetricTracker("loss")

    def train_one_epoch(self) -> float:
        """
        Train one epoch.
        """

        self.model.train()
        self.train_metrics.reset()

        progress = tqdm(
            self.train_loader,
            desc=f"Epoch [{self.current_epoch + 1}/{self.epochs}]",
            dynamic_ncols=True,
            leave=False,
        )

        for batch in progress:

            loss = self.training_step(batch)

            self.train_metrics.update(
                "loss",
                loss,
            )

            progress.set_postfix(
                loss=f"{loss:.4f}"
            )

        return self.train_metrics.average("loss")

    @torch.no_grad()
    def validate(self) -> float:
        """
        Validate for one epoch.
        """

        if self.val_loader is None:
            return 0.0

        self.model.eval()
        self.val_metrics.reset()

        for batch in self.val_loader:

            loss = self.validation_step(batch)

            self.val_metrics.update(
                "loss",
                loss,
            )

        return self.val_metrics.average("loss")

    def fit(self) -> None:
        """
        Complete training loop.
        """

        for epoch in range(self.epochs):

            self.current_epoch = epoch

            train_loss = self.train_one_epoch()

            val_loss = self.validate()

            # if self.scheduler is not None:
            #     self.scheduler.step()
            if self.scheduler is not None:

                if self.scheduler.__class__.__name__ == "ReduceLROnPlateau":

                    self.scheduler.step(val_loss)

                else:

                    self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]

            self.logger.log(
                epoch=epoch + 1,
                train_loss=train_loss,
                val_loss=val_loss,
                learning_rate=current_lr,
            )

            save_last_checkpoint(
                self.checkpoint_dir,
                self.model,
                self.optimizer,
                self.scheduler,
                epoch + 1,
                train_loss,
            )

            if val_loss < self.best_loss:

                self.best_loss = val_loss

                save_best_checkpoint(
                    self.checkpoint_dir,
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch + 1,
                    val_loss,
                )

        self.logger.close()

    def training_step(
        self,
        batch,
    ) -> float:
        """
        Execute a single training step.

        Parameters
        ----------
        batch
            Batch returned by the DataLoader.

        Returns
        -------
        float
            Training loss.
        """

        (view1, view2), _ = batch

        view1 = view1.to(
            self.device,
            non_blocking=True,
        )

        view2 = view2.to(
            self.device,
            non_blocking=True,
        )

        self.optimizer.zero_grad(
            set_to_none=True,
        )

        with autocast(
            device_type="cuda",
            enabled=self.use_amp,
        ):

            loss = self.model.compute_loss(
                view1,
                view2,
            )

        self.scaler.scale(loss).backward()

        if self.config.training.gradient_clip is not None:

            self.scaler.unscale_(self.optimizer)

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.training.gradient_clip,
            )

        self.scaler.step(self.optimizer)

        self.scaler.update()

        self.model.update_target_network()

        return loss.item()

    @torch.no_grad()
    def validation_step(
        self,
        batch,
    ) -> float:
        """
        Execute one validation step.
        """

        (view1, view2), _ = batch

        view1 = view1.to(
            self.device,
            non_blocking=True,
        )

        view2 = view2.to(
            self.device,
            non_blocking=True,
        )

        with autocast(
            device_type="cuda",
            enabled=self.use_amp,
        ):

            loss = self.model.compute_loss(
                view1,
                view2,
            )

        return loss.item()