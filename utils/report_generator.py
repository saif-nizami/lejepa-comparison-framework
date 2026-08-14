"""
Generate a dissertation-ready experiment report.
"""

from __future__ import annotations

import platform
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import yaml

class ReportGenerator:

    def __init__(
        self,
        dataset_name: str,
        output_dir="results",
        config_path="configs/default.yaml",
    ):

        self.output_dir = (
            Path(output_dir)
            / dataset_name.lower()
        )

        self.report_dir = (
            self.output_dir /
            "report"
        )

        self.report_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.config_path = Path(config_path)

        self.summary = pd.read_csv(
            self.output_dir /
            "summary" /
            "experiment_summary.csv"
        )

        with open(self.config_path) as f:

            self.config = yaml.safe_load(f)

    def hardware(self):

        return {

            "OS": platform.platform(),

            "Python": platform.python_version(),

            "PyTorch": torch.__version__,

            "Device": (
                "CUDA"
                if torch.cuda.is_available()
                else (
                    "MPS"
                    if torch.backends.mps.is_available()
                    else "CPU"
                )
            ),

            "Processor": platform.processor(),

        }

    def build_markdown(self):

        hw = self.hardware()

        report = []

        report.append(
            "# Self-Supervised Learning Comparison\n"
        )

        report.append(
            f"Generated: {datetime.now()}\n"
        )

        report.append("---\n")

        report.append("## Dataset\n")

        report.append(
            f"{self.config['dataset']['name']}\n"
        )

        report.append("---\n")

        report.append("## Configuration\n")

        report.append("```yaml\n")

        report.append(
            yaml.dump(
                self.config,
                sort_keys=False,
            )
        )

        report.append("```\n")

        report.append(
            "## Results\n"
        )

        report.append(
            self.summary.to_markdown(index=False)
        )

        report.append("\n")

        report.append(
            "## Hardware\n"
        )

        for k, v in hw.items():

            report.append(
                f"- **{k}** : {v}"
            )

        report.append("\n")

        report.append(
            "## Generated Figures\n"
        )

        report.append(
            "### Comparison Plots\n"
        )

        report.append(
            "![](../plots/comparison/comparison_accuracy.png)\n"
        )

        report.append(
            "![](../plots/comparison/comparison_precision.png)\n"
        )

        report.append(
            "![](../plots/comparison/comparison_recall.png)\n"
        )

        report.append(
            "![](../plots/comparison/comparison_f1.png)\n"
        )

        markdown = "\n".join(report)

        report_file = (
            self.report_dir /
            "report.md"
        )

        with open(
            report_file,
            "w",
            encoding="utf-8",
        ) as f:

            f.write(markdown)

        return report_file

    def archive(self):

        folders = [

            "plots",

            "tables",

            "summary",

            "metrics",

            "confusion_matrix",

            "tsne",

            "umap",

        ]

        appendix = (
            self.report_dir /
            "appendix"
        )

        appendix.mkdir(
            exist_ok=True,
        )

        for folder in folders:

            src = self.output_dir / folder

            if src.exists():

                shutil.copytree(

                    src,

                    appendix / folder,

                    dirs_exist_ok=True,

                )

        shutil.copy(

            self.config_path,

            appendix /
            "default.yaml",

        )

    def generate(self):

        print("=" * 60)

        print(
            "Generating Report..."
        )

        self.build_markdown()

        self.archive()

        print("Done.")

        print("=" * 60)