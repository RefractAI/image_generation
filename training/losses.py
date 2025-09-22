"""Loss aggregation utilities for configurable training loops."""

from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, Mapping

import torch


class LossManager:
    """Track and aggregate named losses with configurable weights."""

    def __init__(self, weights: Mapping[str, float], window: int) -> None:
        if not weights:
            raise ValueError("At least one loss weight must be provided.")
        self.weights: Dict[str, float] = {str(name): float(weight) for name, weight in weights.items()}
        if window <= 0:
            raise ValueError("Averages window must be positive.")
        self.window = int(window)
        self._buffers: Dict[str, deque[float]] = {
            name: deque(maxlen=self.window) for name in self.weights.keys()
        }

    def compute_total(self, losses: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Return the weighted sum of configured losses."""

        missing = [name for name in self.weights.keys() if name not in losses]
        if missing:
            joined = ", ".join(missing)
            raise KeyError(f"Missing loss values for: {joined}")
        total = None
        for name, weight in self.weights.items():
            value = losses[name]
            if not torch.is_tensor(value):
                raise TypeError(f"Loss '{name}' must be a torch.Tensor, got {type(value)!r}.")
            component = value * weight
            total = component if total is None else total + component
        assert total is not None  # for mypy; guarded by weights check
        return total

    def update(self, losses: Mapping[str, torch.Tensor | float]) -> None:
        """Record latest scalar losses for running-window averages."""

        for name, value in losses.items():
            if torch.is_tensor(value):
                tensor = value.detach()
                if tensor.ndim > 0:
                    tensor = tensor.mean()
                numeric = float(tensor.item())
            else:
                numeric = float(value)
            if name not in self._buffers:
                self._buffers[name] = deque(maxlen=self.window)
            self._buffers[name].append(numeric)

    def averages(self) -> Dict[str, float]:
        """Return the mean of each tracked loss over the recent window."""

        return {
            name: (sum(buffer) / len(buffer) if buffer else 0.0)
            for name, buffer in self._buffers.items()
        }

    @property
    def names(self) -> Iterable[str]:
        """A view of currently tracked loss names."""

        return self._buffers.keys()
