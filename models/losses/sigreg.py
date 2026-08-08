"""
SIGReg regularization.

Placeholder implementation.

The complete implementation will be added when
integrating the official LeJEPA repository.
"""

from __future__ import annotations

import torch.nn as nn


class SIGRegLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, loss):

        return loss