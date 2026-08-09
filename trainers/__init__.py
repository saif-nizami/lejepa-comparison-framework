from .trainer import Trainer
from utils.device import get_device
device = get_device

self.device = device
self.config = config
self.use_amp = (
    self.device.type == "cuda"
    and self.config.training.mixed_precision
)

__all__ = [
    "Trainer",
]