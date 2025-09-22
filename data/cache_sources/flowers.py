"""Flowers dataset source for cache generation."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from datasets import load_dataset
from torch.utils.data import Dataset

from config import CacheFlowersDatasetConfig
from data.text_lookup import TextLookupBase

from .common import build_cache_sample

_FLOWERS_LABELS_PATH = Path(__file__).resolve().parent / "labels.txt"

class FlowersCacheDataset(Dataset):
    def __init__(
        self,
        cfg: CacheFlowersDatasetConfig,
        buckets,
        caption_lookup: TextLookupBase | None,
        alt_caption_lookup: TextLookupBase | None,
    ) -> None:
        self.cfg = cfg
        self.dataset = load_dataset(cfg.path, split=cfg.split)
        self.dataset = self.dataset.with_format("python")
        self.buckets = buckets
        self.caption_lookup = caption_lookup
        self.alt_caption_lookup = alt_caption_lookup

        feature = self.dataset.features.get(cfg.label_field)
        lines = [line.strip() for line in _FLOWERS_LABELS_PATH.read_text().splitlines()]
        self.label_map = {idx: line for idx, line in enumerate(lines) if line}

    def __len__(self) -> int:  # pragma: no cover - simple passthrough
        return len(self.dataset)

    def __getitem__(self, index: int):
        example = self.dataset[index]

        image = example.get(self.cfg.image_field)

        key = index
        caption = self.label_map.get(example["label"])

        return build_cache_sample(
            image,
            key,
            self.buckets,
            self.caption_lookup,
            self.alt_caption_lookup,
            caption=caption,
        )


def build_flowers_dataset(
    cfg: CacheFlowersDatasetConfig,
    buckets,
    caption_lookup: TextLookupBase | None,
    alt_caption_lookup: TextLookupBase | None,
):
    dataset = FlowersCacheDataset(cfg, buckets, caption_lookup, alt_caption_lookup)
    return dataset


__all__ = ["build_flowers_dataset"]
