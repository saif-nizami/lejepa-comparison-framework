from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


class ComparisonPlotter:

    def __init__(
        self,
        dataset_name: str,
        output_dir="results",
    ):

        self.output_dir = (
            Path(output_dir)
            / dataset_name.lower()
        )

        self.summary = pd.read_csv(
            self.output_dir /
            "summary" /
            "experiment_summary.csv"
        )

        self.comp_dir = (
            self.output_dir /
            "plots" /
            "comparison"
        )

        self.comp_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def plot_accuracy(self):

        plt.figure(figsize=(10,6))

        plt.bar(

            self.summary["model"],

            self.summary["best_val_accuracy"],

        )

        plt.ylabel("Accuracy (%)")

        plt.title(
            "Linear Probe Accuracy Comparison"
        )

        plt.grid(axis="y")

        plt.tight_layout()

        plt.savefig(

            self.comp_dir /
            "comparison_accuracy.png",

            dpi=600,

        )

        plt.close()

    def plot_precision(self):

        plt.figure(figsize=(10,6))

        plt.bar(

            self.summary["model"],

            self.summary["precision"],

        )

        plt.ylabel("Precision")

        plt.title(
            "Precision Comparison"
        )

        plt.grid(axis="y")

        plt.tight_layout()

        plt.savefig(

            self.comp_dir /
            "comparison_precision.png",

            dpi=600,

        )

        plt.close()

    def plot_recall(self):

        plt.figure(figsize=(10,6))

        plt.bar(

            self.summary["model"],

            self.summary["recall"],

        )

        plt.ylabel("Recall")

        plt.title(
            "Recall Comparison"
        )

        plt.grid(axis="y")

        plt.tight_layout()

        plt.savefig(

            self.comp_dir /
            "comparison_recall.png",

            dpi=600,

        )

        plt.close()

    def plot_f1(self):

        plt.figure(figsize=(10,6))

        plt.bar(

            self.summary["model"],

            self.summary["f1_score"],

        )

        plt.ylabel("F1")

        plt.title(
            "F1-score Comparison"
        )

        plt.grid(axis="y")

        plt.tight_layout()

        plt.savefig(

            self.comp_dir /
            "comparison_f1.png",

            dpi=600,

        )

        plt.close()

    def generate_all(self):

        print("Generating comparison plots...")

        self.plot_accuracy()

        self.plot_precision()

        self.plot_recall()

        self.plot_f1()

        print("Done.")