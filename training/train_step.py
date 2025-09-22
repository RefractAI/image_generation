"""Default training step implementation."""

from __future__ import annotations

import random
from time import perf_counter
from typing import Any, Dict

import torch

from training.loop_context import TrainingLoopContext
from training.rectified_flow import rectified_flow_loss


def default_train_step(
    *, batch: Dict[str, Any], context: TrainingLoopContext, global_step: int
) -> Dict[str, torch.Tensor]:
    """Default training inner loop.

    Returns a dictionary containing at least ``loss``.
    """

    accelerator = context.accelerator
    cfg = context.cfg
    latents = batch[cfg.dataset.latent_field].to(accelerator.device)
    captions = list(batch[cfg.dataset.caption_field])

    alt_field = cfg.dataset.alt_caption_field

    def _extract_alt(values, idx):
        if isinstance(values, torch.Tensor):
            value = values[idx]
            return value.item() if value.ndim == 0 else value
        if isinstance(values, (list, tuple)):
            return values[idx]
        return values

    def _is_valid_candidate(candidate):
        if candidate is None:
            return False
        if isinstance(candidate, str):
            return len(candidate) > cfg.dataset.alt_caption_min_length
        return True

    use_alt_only = cfg.text_encoder.type == "c2i"
    alt_probability = 1.0 if use_alt_only else cfg.dataset.alt_caption_probability

    if alt_field:
        alts = batch.get(alt_field)
        if alts is not None:
            updated = []
            for idx in range(len(captions)):
                candidate = _extract_alt(alts, idx)
                if use_alt_only:
                    if candidate is None:
                        raise ValueError("c2i text encoder requires alt captions for every sample.")
                    updated.append(candidate)
                    continue
                if _is_valid_candidate(candidate) and random.random() < alt_probability:
                    updated.append(candidate)
                else:
                    updated.append(captions[idx])
            captions = updated

    timings = context.performance_timings if context.track_performance else None

    def _sync_if_needed() -> None:
        if timings is not None and torch.cuda.is_available():
            torch.cuda.synchronize()

    if timings is not None:
        _sync_if_needed()
        encode_start = perf_counter()
    cond = context.encode(captions)
    if timings is not None:
        _sync_if_needed()
        timings["text_encode"] = perf_counter() - encode_start
    dropout = cfg.training.cond_dropout
    if dropout > 0:
        if cond.ndim == 3:
            mask = torch.rand(cond.size(0), 1, 1, device=cond.device) < dropout
            cond = torch.where(mask, context.uncond.expand(cond.size(0), -1, -1), cond)
        else:
            mask = torch.rand(cond.size(0), device=cond.device) < dropout
            if mask.any():
                cond = cond.clone()
                fill_value = context.uncond.expand(cond.size(0))
                cond[mask] = fill_value[mask]

    with accelerator.accumulate(context.model):
        context.optimizer.zero_grad(set_to_none=True)
        if timings is not None:
            _sync_if_needed()
            forward_start = perf_counter()
        mse, cf = rectified_flow_loss(context.model, latents, cond)
        losses: Dict[str, torch.Tensor] = {"mse": mse, "cf": cf}
        total_loss = context.loss_manager.compute_total(losses)
        if timings is not None:
            _sync_if_needed()
            timings["forward"] = perf_counter() - forward_start
        losses["loss"] = total_loss
        if timings is not None:
            _sync_if_needed()
            backward_start = perf_counter()
        accelerator.backward(total_loss)
        if timings is not None:
            _sync_if_needed()
            timings["backward"] = perf_counter() - backward_start
        #accelerator.clip_grad_norm_(context.model.parameters(), cfg.training.clip_grad)
        if timings is not None:
            _sync_if_needed()
            optimizer_start = perf_counter()
        context.optimizer.step()
        context.scheduler.step()
        if timings is not None:
            _sync_if_needed()
            timings["optimizer"] = perf_counter() - optimizer_start

    return losses
