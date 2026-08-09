"""
Training metrics utilities.

Provides running averages for losses and other metrics.
"""

from __future__ import annotations


class AverageMeter:
    """
    Computes and stores the running average of a metric.

    Example
    -------
    >>> loss_meter = AverageMeter()

    >>> loss_meter.update(0.52)

    >>> loss_meter.update(0.41)

    >>> print(loss_meter.average)
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset all statistics."""

        self.value = 0.0
        self.sum = 0.0
        self.count = 0
        self.average = 0.0

    def update(
        self,
        value: float,
        n: int = 1,
    ) -> None:
        """
        Update running statistics.

        Parameters
        ----------
        value : float
            New metric value.

        n : int
            Number of samples represented by value.
        """

        self.value = value
        self.sum += value * n
        self.count += n

        self.average = self.sum / self.count

    def __str__(self) -> str:
        return f"{self.average:.4f}"

class MetricTracker:
    """
    Tracks multiple metrics simultaneously.
    """

    def __init__(self, *names: str) -> None:

        self.metrics = {
            name: AverageMeter()
            for name in names
        }

    def reset(self) -> None:

        for metric in self.metrics.values():
            metric.reset()

    def update(
        self,
        name: str,
        value: float,
        n: int = 1,
    ) -> None:

        self.metrics[name].update(value, n)

    def average(
        self,
        name: str,
    ) -> float:

        return self.metrics[name].average

    def as_dict(self) -> dict:

        return {
            name: meter.average
            for name, meter in self.metrics.items()
        }