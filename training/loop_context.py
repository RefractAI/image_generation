"""Shared context container for configurable training loops."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from accelerate import Accelerator
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
import torch

from config import TrainConfig
from models.vae import BaseVAE
from training.losses import LossManager


@dataclass
class TrainingLoopContext:
    accelerator: Accelerator
    model: torch.nn.Module
    optimizer: Optimizer
    scheduler: object
    encode: Callable[[list[str]], torch.Tensor]
    uncond: torch.Tensor
    cfg: TrainConfig
    loss_manager: LossManager
    vae: Optional[BaseVAE] = None
    samples_dir: Optional[Path] = None
    writer: Optional[SummaryWriter] = None
    track_performance: bool = False
    performance_timings: dict[str, float] = field(default_factory=dict)
    fixed_images: Optional[torch.Tensor] = None
    fixed_labels: Optional[torch.Tensor] = None
    fixed_noise: Optional[torch.Tensor] = None
    global_step: int = 0
