"""
Experiment Logger

Automatically stores experiment results for all SSL models.

Outputs
-------
results/

    summary/
        experiment_summary.csv
        experiment_summary.xlsx
        experiment_summary.json

    metrics/

        simclr/
            simclr_001_20260815_142011.json
            ...

        byol/
        vicreg/
        barlow_twins/
        lejepa/
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd


class ExperimentLogger:

    def __init__(
        self,
        dataset_name: str,
        output_dir: str = "results",
    ) -> None:

        self.output_dir = (
            Path(output_dir)
            / dataset_name.lower()
        )

        self.summary_dir = (
            self.output_dir / "summary"
        )

        self.metrics_dir = (
            self.output_dir / "metrics"
        )

        self.summary_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.metrics_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.csv_path = (
            self.summary_dir /
            "experiment_summary.csv"
        )

        self.xlsx_path = (
            self.summary_dir /
            "experiment_summary.xlsx"
        )

        self.json_path = (
            self.summary_dir /
            "experiment_summary.json"
        )

    # ==========================================================
    # Run Number
    # ==========================================================

    def _next_run_number(
        self,
        model: str,
    ) -> tuple[int, Path]:

        model = model.lower()

        model_dir = (
            self.metrics_dir /
            model
        )

        model_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        existing = list(
            model_dir.glob("*.json")
        )

        if len(existing) == 0:
            return 1, model_dir

        runs = []

        for file in existing:

            match = re.search(
                rf"{model}_(\d+)_",
                file.stem,
            )

            if match:

                runs.append(
                    int(match.group(1))
                )

        if len(runs) == 0:
            return 1, model_dir

        return max(runs) + 1, model_dir

    # ==========================================================
    # Public API
    # ==========================================================

    def log(
        self,
        history=None,
        **kwargs,
    ) -> None:

        if "model" not in kwargs:

            raise ValueError(
                "'model' is required."
            )

        model = kwargs["model"].lower()

        run_number, model_dir = (
            self._next_run_number(
                model
            )
        )

        timestamp = datetime.now()

        timestamp_string = timestamp.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        filename_timestamp = timestamp.strftime(
            "%Y%m%d_%H%M%S"
        )

        experiment_id = (
            f"{model.upper()}_{run_number:03d}"
        )

        row = {

            "experiment_id": experiment_id,

            "timestamp": timestamp_string,

            **kwargs,
        }

        # ======================================================
        # Save Individual JSON
        # ======================================================

        json_file = (
            model_dir /
            f"{model}_{run_number:03d}_{filename_timestamp}.json"
        )

        with open(
            json_file,
            "w",
        ) as f:

            json.dump(
                row,
                f,
                indent=4,
            )

        # ======================================================
        # Save Training History
        # ======================================================

        if history is not None:

            history_file = (
                model_dir /
                f"{model}_{run_number:03d}_{filename_timestamp}_history.csv"
            )

            import pandas as pd

            pd.DataFrame(history).to_csv(
                history_file,
                index=False,
            )

        # ======================================================
        # Update CSV / Excel
        # ======================================================

        if self.csv_path.exists():

            df = pd.read_csv(
                self.csv_path
            )

        else:

            df = pd.DataFrame()

        df = pd.concat(
            [
                df,
                pd.DataFrame([row]),
            ],
            ignore_index=True,
        )

        df.to_csv(
            self.csv_path,
            index=False,
        )

        df.to_excel(
            self.xlsx_path,
            index=False,
        )

        with open(
            self.json_path,
            "w",
        ) as f:

            json.dump(
                df.to_dict(
                    orient="records",
                ),
                f,
                indent=4,
            )

        print("=" * 60)
        print("Experiment Saved")
        print("=" * 60)
        print("Experiment :", experiment_id)
        print("Model      :", model)
        print("Timestamp  :", timestamp_string)
        print("JSON       :", json_file)
        print("CSV        :", self.csv_path)
        print("Excel      :", self.xlsx_path)
        print("=" * 60)

        return {
            "history_file": history_file if history is not None else None,
            "json_file": json_file,
        }