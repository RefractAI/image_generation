"""Aspect-ratio bucketing utilities.

This module provides a simple bucket-aware batch sampler that works across
single- and multi-GPU training. It keeps the implementation intentionally
minimal: buckets are registered once, batches are reshuffled each epoch, and
ranks consume disjoint batches when distributed training is active.
"""

from __future__ import annotations

import math
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Tuple, TYPE_CHECKING

import torch
from torch.utils.data import Sampler

from .aspect_ratios import (
    aspect_ratio_buckets_1024,
    aspect_ratio_buckets_512,
    aspect_ratio_buckets_256,
    bucket_spec,
)

if TYPE_CHECKING:
    from config import TrainConfig
    from training.loop_context import TrainingLoopContext


BUCKET_CACHE_DIR = Path(".bucket_cache")


def _world_info() -> Tuple[int, int]:
    """Return ``(world_size, rank)`` even if torch.distributed is unavailable."""

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size(), torch.distributed.get_rank()
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    return world or 1, rank


class BucketBatchSampler(Sampler[List[int]]):
    """Bucket-aware batch sampler with deterministic epoch shuffling.

    The sampler expects ``dataset[i]`` to provide a ``"target_resolution"`` field
    describing the latent resolution (width, height) used for bucketing.
    """

    def __init__(
        self,
        dataset,
        aspect_ratio_buckets: Sequence[Tuple[int, int]],
        batch_size: int,
        drop_last: bool = True,
        base_seed: int = 0,
        dataset_id: str | None = None,
    ) -> None:
        super().__init__(dataset)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.base_seed = base_seed
        self.epoch = 0
        self.generator = torch.Generator(device="cpu")
        self.dataset_id = dataset_id
        self.cache_file = None

        world, rank = _world_info()
        self.world_size = max(world, 1)
        self.rank = max(rank, 0)

        if self.dataset_id:
            BUCKET_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            safe_id = re.sub(r"[^a-zA-Z0-9._-]", "_", self.dataset_id)
            self.cache_file = BUCKET_CACHE_DIR / f"{safe_id}_bs{self.batch_size}.pt"

        self._buckets: Dict[Tuple[int, int], List[int]] = {tuple(res): [] for res in aspect_ratio_buckets}
        if not self._load_bucket_cache():
            self._populate_buckets()
            self._maybe_save_bucket_cache()
        self._num_batches = self._estimate_total_batches()

    # --------------------------------------------------------------------- utils
    def _populate_buckets(self) -> None:
        if "target_resolution" not in self.dataset.column_names:
            raise KeyError("Dataset must provide a 'target_resolution' column for bucketing.")

        if self.rank == 0:
            label = self.dataset_id or "dataset"
            print(
                f"[BucketBatchSampler] Fetching 'target_resolution' column for {label!r}."
                " This may take a moment on first use."
            )

        resolutions = self.dataset["target_resolution"]
        if isinstance(resolutions, list):
            iterator = resolutions
        else:
            iterator = list(resolutions)

        total = len(self.dataset) if hasattr(self.dataset, "__len__") else len(iterator)
        start = time.time()
        if self.rank == 0:
            label = self.dataset_id or "dataset"
            print(
                f"[BucketBatchSampler] Building bucket index for {label!r} "
                f"(batch_size={self.batch_size}, samples={total:,})."
            )
        progress_step = max(total // 10, 1)
        for idx, res_value in enumerate(iterator):
            res_tuple = tuple(int(v) for v in (res_value.tolist() if hasattr(res_value, "tolist") else res_value))
            if res_tuple in self._buckets:
                self._buckets[res_tuple].append(idx)

        if self.rank == 0:
            elapsed = time.time() - start
            print(
                f"[BucketBatchSampler] Bucket index ready in {elapsed:.1f}s for {self.dataset_id or 'dataset'}."
            )

    def _estimate_total_batches(self) -> int:
        total = 0
        for indices in self._buckets.values():
            if not indices:
                continue
            full, remainder = divmod(len(indices), self.batch_size)
            total += full
            if remainder and not self.drop_last:
                total += 1
        return total

    def _seed(self) -> None:
        self.generator.manual_seed(self.base_seed + self.epoch)
        random.seed(self.base_seed + self.epoch)

    def _build_batches(self) -> List[List[int]]:
        self._seed()
        batches: List[List[int]] = []
        for indices in self._buckets.values():
            if not indices:
                continue
            shuffled = [indices[i] for i in torch.randperm(len(indices), generator=self.generator).tolist()]
            for start in range(0, len(shuffled), self.batch_size):
                batch = shuffled[start : start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                if len(batch) < self.batch_size and not self.drop_last:
                    # pad by wrapping around the shuffled pool
                    pad_from = self.batch_size - len(batch)
                    batch.extend(shuffled[:pad_from])
                batches.append(batch)
        random.shuffle(batches)
        return batches

    def _partition_batches(self, batches: List[List[int]]) -> List[List[int]]:
        if self.world_size == 1:
            return batches
        if not batches:
            return []
        if self.drop_last:
            usable = len(batches) - (len(batches) % self.world_size)
            batches = batches[:usable]
        else:
            remainder = (-len(batches)) % self.world_size
            if remainder and batches:
                for idx in range(remainder):
                    batches.append(list(batches[idx % len(batches)]))
        return batches

    # ---------------------------------------------------------------- interface
    def __iter__(self) -> Iterator[List[int]]:
        batches = self._partition_batches(self._build_batches())
        if not batches:
            return
        start = self.rank % self.world_size
        for batch in batches[start::self.world_size]:
            yield batch

    def __len__(self) -> int:
        if self.world_size == 1:
            return self._num_batches
        if self.drop_last:
            usable = self._num_batches - (self._num_batches % self.world_size)
            return usable // self.world_size
        return math.ceil(self._num_batches / self.world_size)

    # ``DistributedSampler`` API compatibility so Accelerate can advance epochs.
    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def indices_for_resolution(self, resolution: Tuple[int, int]) -> List[int]:
        """Return dataset indices registered for the given resolution."""

        return self._buckets.get(tuple(int(v) for v in resolution), [])

    # ---------------------------------------------------------------- cache utils
    def _load_bucket_cache(self) -> bool:
        if not self.cache_file or not self.cache_file.exists():
            return False
        try:
            data = torch.load(self.cache_file)
        except Exception as exc:  # pragma: no cover - cache fallback
            if self.rank == 0:
                print(
                    f"[BucketBatchSampler] Failed to load cache {self.cache_file}: {exc}. Rebuilding."
                )
            return False
        for key, indices in data.items():
            key_tuple = tuple(int(v) for v in key)
            self._buckets[key_tuple] = list(indices)
        if self.rank == 0:
            total = sum(len(v) for v in self._buckets.values())
            print(
                f"[BucketBatchSampler] Loaded bucket cache {self.cache_file} "
                f"({total:,} assignments)."
            )
        return True

    def _maybe_save_bucket_cache(self) -> None:
        if not self.cache_file or self.rank != 0:
            return
        tmp_path = self.cache_file.with_suffix(self.cache_file.suffix + ".tmp")
        data = {tuple(int(v) for v in key): list(indices) for key, indices in self._buckets.items()}
        torch.save(data, tmp_path)
        tmp_path.replace(self.cache_file)
        total = sum(len(v) for v in self._buckets.values())
        print(
            f"[BucketBatchSampler] Saved bucket cache {self.cache_file} "
            f"({total:,} assignments)."
        )


def get_sample_batch(
    dataset,
    dataloader,
    bucket_batch_sampler: BucketBatchSampler | None,
    *,
    context: "TrainingLoopContext",
    cfg: "TrainConfig",
):
    """Return fixed latents, text embeddings, and noise for logging samples."""

    if context.vae is None:
        raise ValueError("Training loop context must provide a VAE for sampling.")

    accelerator = context.accelerator
    vae = context.vae
    encode_text_fn = context.encode

    dataset_cfg = cfg.dataset
    sampling_cfg = cfg.sampling
    model_cfg = cfg.model

    caption_column = dataset_cfg.caption_field
    alt_caption_column = dataset_cfg.alt_caption_field
    latent_column = dataset_cfg.latent_field
    alt_probability = dataset_cfg.alt_caption_probability
    alt_min_length = dataset_cfg.alt_caption_min_length

    _, multires_sample_res, single_sample_res = bucket_spec(dataset_cfg.image_size)
    sample_res = multires_sample_res if dataset_cfg.multires else single_sample_res
    target_res = tuple(int(v) for v in sample_res)
    num_samples = min(cfg.sampling.num_samples, cfg.dataset.batch_size)

    vae_device = accelerator.device
    if hasattr(vae, "model"):
        try:
            vae_device = next(vae.model.parameters()).device
        except (StopIteration, AttributeError):
            pass

    if dataset_cfg.multires:
        if bucket_batch_sampler is None:
            raise ValueError("Bucket batch sampler is required when multires is enabled.")
        bucket = bucket_batch_sampler.indices_for_resolution(target_res)
        if not bucket:
            raise ValueError(f"No samples available for bucket {target_res}.")
        chosen = bucket[: min(num_samples, len(bucket))]
        batch = [dataset[i] for i in chosen]
    else:
        batch = next(iter(dataloader))
        latents = batch[latent_column]
        chosen = list(range(min(num_samples, len(latents))))

    def _stack_latents():
        if dataset_cfg.multires:
            latents = torch.stack([item[latent_column] for item in batch])
        else:
            latents = batch[latent_column][chosen]
        return latents.to(vae_device)

    fixed_latents = _stack_latents()
    with torch.no_grad():
        fixed_images = vae.decode(fixed_latents).to(accelerator.device)

    def _choose_caption(base: str, candidate: str | None) -> str:
        if isinstance(candidate, str) and len(candidate) >= alt_min_length and random.random() < alt_probability:
            return candidate
        return base

    captions: List[str] = []
    if dataset_cfg.multires:
        for example in batch:
            base = example.get(caption_column, "")
            candidate = example.get(alt_caption_column)
            captions.append(_choose_caption(base, candidate))
    else:
        raw_captions = batch.get(caption_column)
        alt_captions = batch.get(alt_caption_column)
        for idx in chosen:
            base = ""
            if isinstance(raw_captions, (list, tuple)) and idx < len(raw_captions):
                base = raw_captions[idx]
            elif isinstance(raw_captions, str):
                base = raw_captions
            candidate: str | None = None
            if isinstance(alt_captions, (list, tuple)) and idx < len(alt_captions):
                candidate = alt_captions[idx]
            elif isinstance(alt_captions, str):
                candidate = alt_captions
            captions.append(_choose_caption(base, candidate))

    with torch.no_grad():
        fixed_labels = encode_text_fn(captions)
        latent_h = target_res[1] // dataset_cfg.scale_factor
        latent_w = target_res[0] // dataset_cfg.scale_factor
        fixed_noise = torch.randn(
            len(captions),
            model_cfg.channels,
            latent_h,
            latent_w,
            device=accelerator.device,
        )

    fixed_labels.original_captions = captions

    return fixed_images, fixed_labels, fixed_noise
