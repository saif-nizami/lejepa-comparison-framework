"""
Generate dissertation-ready comparison tables.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class TableGenerator:

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

        self.table_dir = (
            self.output_dir /
            "tables"
        )

        self.table_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def generate_accuracy_table(self):

        table = self.summary[[
            "model",
            "dataset",
            "best_val_accuracy",
            "precision",
            "recall",
            "f1_score",
        ]]

        table = table.sort_values(
            "best_val_accuracy",
            ascending=False,
        )

        table.insert(
            0,
            "Rank",
            range(
                1,
                len(table) + 1,
            ),
        )

        table.to_csv(
            self.table_dir /
            "accuracy_comparison.csv",
            index=False,
        )

        table.to_excel(
            self.table_dir /
            "accuracy_comparison.xlsx",
            index=False,
        )

        table.to_latex(
            self.table_dir /
            "accuracy_comparison.tex",
            index=False,
            float_format="%.4f",
        )

        return table

    def generate_speed_table(self):

        cols = []

        for c in [
            "model",
            "ssl_training_time",
            "probe_training_time",
            "inference_time",
        ]:
            if c in self.summary.columns:
                cols.append(c)

        if len(cols) <= 1:
            return

        table = self.summary[cols]

        table.to_csv(
            self.table_dir /
            "speed_comparison.csv",
            index=False,
        )

        table.to_excel(
            self.table_dir /
            "speed_comparison.xlsx",
            index=False,
        )

    def generate_parameter_table(self):

        cols = []

        for c in [
            "model",
            "backbone_parameters",
            "classifier_parameters",
            "total_parameters",
        ]:
            if c in self.summary.columns:
                cols.append(c)

        if len(cols) <= 1:
            return

        table = self.summary[cols]

        table.to_csv(
            self.table_dir /
            "parameter_comparison.csv",
            index=False,
        )

        table.to_excel(
            self.table_dir /
            "parameter_comparison.xlsx",
            index=False,
        )

    def generate_all(self):

        print("=" * 60)
        print("Generating comparison tables")
        print("=" * 60)

        self.generate_accuracy_table()
        self.generate_speed_table()
        self.generate_parameter_table()

        print("Done.")