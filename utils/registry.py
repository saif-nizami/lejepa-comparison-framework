"""
Registry for SSL models.
"""

from __future__ import annotations

SSL_MODELS: dict[str, type] = {}


def register_ssl_model(name: str):

    def decorator(cls):

        key = name.lower()

        if key in SSL_MODELS:
            raise ValueError(
                f"'{key}' already registered."
            )

        SSL_MODELS[key] = cls

        return cls

    return decorator


def get_ssl_model(name: str):

    key = name.lower()

    if key not in SSL_MODELS:

        raise ValueError(
            f"Unknown SSL model '{name}'. "
            f"Available models: {list_models()}"
        )

    return SSL_MODELS[key]


def list_models():

    return sorted(SSL_MODELS.keys())