"""
Run all SSL experiments sequentially.
"""

from __future__ import annotations

import subprocess
import sys

MODELS = [
    "simclr",
    "byol",
    "vicreg",
    "barlow_twins",
    "lejepa",
]


def run(command: list[str]) -> None:

    print("=" * 80)
    print(" ".join(command))
    print("=" * 80)

    subprocess.run(
        command,
        check=True,
    )


def main():

    for model in MODELS:

        print(f"\nRunning {model}\n")

        # --------------------------------------------------
        # Train SSL
        # --------------------------------------------------

        run(
            [
                sys.executable,
                "scripts/train.py",
                "--config",
                "configs/default.yaml",
                "--model",
                model,
            ]
        )

        # --------------------------------------------------
        # Linear Probe
        # --------------------------------------------------

        run(
            [
                sys.executable,
                "scripts/linear_probe.py",
                "--config",
                "configs/default.yaml",
                "--model",
                model,
            ]
        )

        # --------------------------------------------------
        # Evaluate
        # --------------------------------------------------

        run(
            [
                sys.executable,
                "scripts/evaluate.py",
                "--config",
                "configs/default.yaml",
                "--model",
                model,
            ]
        )


if __name__ == "__main__":
    main()