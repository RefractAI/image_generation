"""Rectified flow helpers shared across training and inference."""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def rectified_flow_loss(
    model: torch.nn.Module,
    x: torch.Tensor,
    cond: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute rectified flow loss terms for the given batch."""

    batch = x.size(0)
    device = x.device
    dtype = x.dtype
    t = torch.sigmoid(torch.randn((batch,), device=device, dtype=dtype))
    shape = [batch] + [1] * (x.dim() - 1)
    z1 = torch.randn_like(x)
    zt = (1 - t.view(shape)) * x + t.view(shape) * z1
    velocity = model(zt.contiguous(), t.contiguous(), cond.contiguous())
    target = z1 - x
    reduce_dims = tuple(range(1, x.dim()))
    mse = ((target - velocity) ** 2).mean(dim=reduce_dims)
    cf = ((velocity - torch.roll(target, shifts=1, dims=0)) ** 2).mean(dim=reduce_dims)
    return mse.mean(), cf.mean()


def rectified_flow_sample(
    model: torch.nn.Module,
    z: torch.Tensor,
    cond: torch.Tensor,
    null_cond: Optional[torch.Tensor],
    steps: int,
    guidance_scale: float,
) -> torch.Tensor:
    """Reverse integrate rectified flow velocities to sample latents."""

    latents = z.clone()
    batch = latents.size(0)
    device = latents.device
    dtype = latents.dtype
    view_shape = [batch] + [1] * (latents.dim() - 1)
    dt = torch.full((batch,), 1.0 / steps, device=device, dtype=dtype).view(view_shape)
    for idx in range(steps, 0, -1):
        t = torch.full((batch,), idx / steps, device=device, dtype=dtype)
        guided = model(latents, t, cond)
        if null_cond is not None:
            unguided = model(latents, t, null_cond)
            guided = unguided + guidance_scale * (guided - unguided)
        latents = latents - dt * guided
    return latents
