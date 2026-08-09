"""
Logging utilities for SSL training.

Supports:
- Console logging
- CSV logging
- TensorBoard logging
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


class TrainingLogger:
    """
    Training logger.

    Logs metrics to:
        - Console
        - CSV
        - TensorBoard
    """

    def __init__(
        self,
        log_dir: str | Path,
        experiment_name: str,
        use_tensorboard: bool = True,
        use_csv: bool = True,
    ) -> None:

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.log_dir = (
            Path(log_dir)
            / f"{experiment_name}_{timestamp}"
        )

        self.log_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.use_tensorboard = use_tensorboard
        self.use_csv = use_csv

        # --------------------------------------------------
        # TensorBoard
        # --------------------------------------------------

        self.writer = None

        if self.use_tensorboard:
            self.writer = SummaryWriter(
                log_dir=self.log_dir
            )

        # --------------------------------------------------
        # CSV
        # --------------------------------------------------

        self.csv_file = None
        self.csv_writer = None

        if self.use_csv:

            self.csv_file = open(
                self.log_dir / "training_log.csv",
                "w",
                newline="",
                encoding="utf-8",
            )

            self.csv_writer = csv.writer(
                self.csv_file
            )

            self.csv_writer.writerow(
                [
                    "epoch",
                    "train_loss",
                    "val_loss",
                    "learning_rate",
                ]
            )

    # ======================================================
    # Log
    # ======================================================

    def log(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        learning_rate: float,
    ) -> None:

        # Console

        print(
            f"Epoch {epoch:03d} | "
            f"Train {train_loss:.4f} | "
            f"Val {val_loss:.4f} | "
            f"LR {learning_rate:.6f}"
        )

        # TensorBoard

        if self.writer is not None:

            self.writer.add_scalar(
                "Loss/Train",
                train_loss,
                epoch,
            )

            self.writer.add_scalar(
                "Loss/Validation",
                val_loss,
                epoch,
            )

            self.writer.add_scalar(
                "LearningRate",
                learning_rate,
                epoch,
            )

        # CSV

        if self.csv_writer is not None:

            self.csv_writer.writerow(
                [
                    epoch,
                    train_loss,
                    val_loss,
                    learning_rate,
                ]
            )

            self.csv_file.flush()

    # ======================================================
    # Close
    # ======================================================

    def close(self) -> None:

        if self.writer is not None:
            self.writer.close()

        if self.csv_file is not None:
            self.csv_file.close()