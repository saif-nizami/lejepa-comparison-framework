"""
Configuration utilities.

Loads YAML configuration files and provides
attribute-style access to configuration values.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """
    Dictionary with attribute-style access.

    Example
    -------
    cfg.training.epochs
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = Config(value)

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(
                f"No configuration field named '{name}'."
            ) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def load_config(config_path: str | Path) -> Config:
    """
    Load a YAML configuration file.

    Parameters
    ----------
    config_path : str | Path
        Path to the YAML configuration.

    Returns
    -------
    Config
        Parsed configuration object.
    """

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return Config(config)


def save_config(
    config: Config,
    output_path: str | Path,
) -> None:
    """
    Save configuration to YAML.
    """

    output_path = Path(output_path)

    with open(output_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(
            dict(config),
            file,
            sort_keys=False,
        )