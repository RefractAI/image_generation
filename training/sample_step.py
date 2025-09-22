"""Default sampling step implementation."""

from __future__ import annotations

import torch
from torchvision.utils import make_grid, save_image

from training.loop_context import TrainingLoopContext
from training.rectified_flow import rectified_flow_sample


def default_sample_step(*, context: TrainingLoopContext, global_step: int) -> None:
    """Default sampling inner loop triggered at configured intervals."""

    

    accelerator = context.accelerator
    cfg = context.cfg
    model = context.model
    if context.vae is None or context.samples_dir is None:
        raise ValueError("Sample step requires 'vae' and 'samples_dir' in context.")
    if context.fixed_labels is None or context.fixed_noise is None or context.fixed_images is None:
        raise ValueError("Sample step requires precomputed fixed labels, noise, and images.")
    num_samples = min(cfg.sampling.num_samples, cfg.dataset.batch_size)

    model.eval()
    with torch.no_grad():
        cond = context.fixed_labels.to(accelerator.device)
        noise = context.fixed_noise.to(accelerator.device)
        uncond = context.uncond.to(accelerator.device)
        if cond.ndim == 3:
            null_cond = uncond.expand(cond.size(0), -1, -1)
        else:
            null_cond = uncond.expand(cond.size(0))
        sampled = rectified_flow_sample(
            context.model,
            noise,
            cond,
            null_cond,
            cfg.sampling.steps,
            cfg.sampling.cfg,
        )
        decoded = context.vae.decode(sampled.to(context.vae.model.device)).to(accelerator.device)
        stack = torch.cat([context.fixed_images, decoded], dim=0)
        grid = make_grid((stack * 0.5 + 0.5).clamp(0, 1), nrow=num_samples).cpu()
        output_path = context.samples_dir / f"sample_{global_step}.png"
        save_image(grid, output_path, normalize=False)
        if context.writer:
            context.writer.add_image("samples", grid, global_step)
    model.train()
