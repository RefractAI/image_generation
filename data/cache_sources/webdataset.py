"""WebDataset source for cache generation."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Iterable

import webdataset as wds

from config import CacheShardConfig
from data.text_lookup import TextLookupBase

from .common import build_cache_sample


def _format_shard_name(pattern: str, index: int) -> str:
    if "%" in pattern:
        return pattern % index
    if "{" in pattern:
        return pattern.format(index=index)
    return pattern


def collect_shard_files(cfg: CacheShardConfig) -> list[str]:
    shard_dir = Path(cfg.directory)
    if not shard_dir.exists():
        raise FileNotFoundError(f"Shard directory {shard_dir} not found")

    files: list[str] = []
    for shard_idx in range(cfg.min_index, cfg.max_index + 1):
        shard_name = _format_shard_name(cfg.pattern, shard_idx)
        shard_path = shard_dir / shard_name
        if shard_path.exists():
            files.append(str(shard_path))

    if not files:
        raise ValueError(
            f"No shards found in {shard_dir} using pattern {cfg.pattern} "
            f"between indexes {cfg.min_index} and {cfg.max_index}"
        )

    return files


def _preprocess(
    sample,
    buckets: Iterable[tuple[int, int]],
    caption_lookup: TextLookupBase | None,
    alt_caption_lookup: TextLookupBase | None,
):
    image, key = sample
    return build_cache_sample(image, key, buckets, caption_lookup, alt_caption_lookup)


def build_webdataset_dataset(
    cfg: CacheShardConfig,
    buckets: Iterable[tuple[int, int]],
    caption_lookup: TextLookupBase | None,
    alt_caption_lookup: TextLookupBase | None,
):
    shard_files = collect_shard_files(cfg)
    preprocess_fn = partial(
        _preprocess, buckets=buckets, caption_lookup=caption_lookup, alt_caption_lookup=alt_caption_lookup
    )
    dataset = wds.WebDataset(shard_files).decode("pil").to_tuple("webp", "__key__")
    dataset = dataset.map(preprocess_fn)
    return dataset


__all__ = ["build_webdataset_dataset", "collect_shard_files"]
