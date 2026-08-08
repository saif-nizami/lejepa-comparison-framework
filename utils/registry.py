"""
Registry for SSL models.
"""

from __future__ import annotations

SSL_MODELS = {}


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


def list_models():

    return sorted(SSL_MODELS.keys())