from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from datetime import datetime


class ResultsLogger:
    """
    Saves experiment history and metrics.
    """

    def __init__(
        self,
        output_dir: str | Path,
        dataset: str,
        model: str,
    ) -> None:

        # self.output_dir = (
        #     Path(output_dir)
        #     / dataset.lower()
        #     / model.lower()
        # )

        base_dir = (
            Path(output_dir)
            / dataset.lower()
            / model.lower()
        )

        base_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        existing_runs = sorted(
            p for p in base_dir.iterdir()
            if p.is_dir() and p.name.startswith("run_")
        )

        run_index = len(existing_runs) + 1

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        self.output_dir = (
            base_dir
            / f"run_{run_index:03d}_{timestamp}"
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.history = []

    # =====================================================

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        learning_rate: float,
    ) -> None:

        self.history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": learning_rate,
            }
        )

    # =====================================================

    def save_history(self) -> None:

        df = pd.DataFrame(self.history)

        df.to_csv(
            self.output_dir / "training_history.csv",
            index=False,
        )

    # =====================================================

    def save_metrics(
        self,
        metrics: dict,
    ) -> None:

        with open(
            self.output_dir / "metrics.json",
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                metrics,
                file,
                indent=4,
            )

    # =====================================================

    def copy_config(
        self,
        config_path: str | Path,
    ) -> None:

        shutil.copy(
            config_path,
            self.output_dir / "config.yaml",
        )