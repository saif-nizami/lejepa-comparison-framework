"""
Negative Cosine Similarity Loss (BYOL)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .utils import cosine_similarity


class NegativeCosineSimilarity(nn.Module):

    def forward(self, p, z):

        return 2 - 2 * cosine_similarity(
            p,
            z.detach(),
        ).mean()