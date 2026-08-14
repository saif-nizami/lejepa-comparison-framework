"""
Training plot generator.

Creates:

- Loss curves
- Accuracy curves
- Learning-rate curves
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


class Plotter:

    def __init__(
        self,
        model_name: str,
        dataset_name,
        history_file,
        output_dir: str = "results",
    ):

        self.model_name = model_name.lower()

        self.history = pd.read_csv(
            history_file,
        )

        self.output_dir = (
            Path(output_dir)
            / dataset_name.lower()
        )

        self.loss_dir = (
            self.output_dir /
            "plots" /
            "loss"
        )

        self.acc_dir = (
            self.output_dir /
            "plots" /
            "accuracy"
        )

        self.lr_dir = (
            self.output_dir /
            "plots" /
            "learning_rate"
        )

        self.loss_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.acc_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.lr_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def plot_loss(self):

        plt.figure(figsize=(8, 6))

        plt.plot(
            self.history["epoch"],
            self.history["train_loss"],
            label="Train",
            linewidth=2,
        )

        plt.plot(
            self.history["epoch"],
            self.history["val_loss"],
            label="Validation",
            linewidth=2,
        )

        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        plt.title(
            f"{self.model_name.upper()} Loss"
        )

        plt.grid(True)

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            self.loss_dir /
            f"{self.model_name}_loss.png",
            dpi=300,
        )

        plt.close()

    def plot_accuracy(self):

        plt.figure(figsize=(8,6))

        plt.plot(

            self.history["epoch"],

            self.history["train_accuracy"],

            label="Train",

            linewidth=2,

        )

        plt.plot(

            self.history["epoch"],

            self.history["val_accuracy"],

            label="Validation",

            linewidth=2,

        )

        plt.xlabel("Epoch")

        plt.ylabel("Accuracy (%)")

        plt.title(
            f"{self.model_name.upper()} Accuracy"
        )

        plt.grid(True)

        plt.legend()

        plt.tight_layout()

        plt.savefig(

            self.acc_dir /
            f"{self.model_name}_accuracy.png",

            dpi=300,

        )

        plt.close()

    def plot_learning_rate(self):

        if "learning_rate" not in self.history.columns:
            return

        plt.figure(figsize=(8,6))

        plt.plot(

            self.history["epoch"],

            self.history["learning_rate"],

            linewidth=2,

        )

        plt.xlabel("Epoch")

        plt.ylabel("Learning Rate")

        plt.title(
            f"{self.model_name.upper()} Learning Rate"
        )

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(

            self.lr_dir /
            f"{self.model_name}_learning_rate.png",

            dpi=300,

        )

        plt.close()

    def generate_all(self):

        print("Generating plots...")

        self.plot_loss()

        self.plot_accuracy()

        self.plot_learning_rate()

        print("Plots saved.")