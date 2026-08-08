"""
Model factory.
"""

from __future__ import annotations

from utils.registry import SSL_MODELS


def build_ssl_model(
    method: str,
    **kwargs,
):
    """
    Build an SSL model.

    Example
    -------
    model = build_ssl_model(
        method="simclr",
        projection_dim=2048,
        temperature=0.5,
    )
    """

    method = method.lower()

    if method not in SSL_MODELS:
        raise ValueError(
            f"Unknown SSL method '{method}'. "
            f"Available: {sorted(SSL_MODELS.keys())}"
        )

    return SSL_MODELS[method](**kwargs)