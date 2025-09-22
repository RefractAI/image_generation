"""DenseFusion JSONL source for cache generation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset

from config import CacheDenseFusionDatasetConfig
from data.text_lookup import TextLookupBase

from .common import build_cache_sample


class DenseFusionCacheDataset(Dataset):
    def __init__(
        self,
        cfg: CacheDenseFusionDatasetConfig,
        buckets: Iterable[tuple[int, int]],
        caption_lookup: TextLookupBase | None,
        alt_caption_lookup: TextLookupBase | None,
    ) -> None:
        dataset = load_dataset("json", data_files=cfg.jsonl_path)
        if hasattr(dataset, "values") and callable(getattr(dataset, "values", None)):
            dataset = next(iter(dataset.values()))
        dataset = dataset.with_format("python")

        self.dataset = dataset
        self.cfg = cfg
        self.buckets = buckets
        self.caption_lookup = caption_lookup
        self.alt_caption_lookup = alt_caption_lookup
        self.image_root = Path(cfg.image_root) if cfg.image_root else Path(cfg.jsonl_path).resolve().parent

    def __len__(self) -> int:  # pragma: no cover - passthrough
        return len(self.dataset)

    def __getitem__(self, index: int):
        example = self.dataset[index]
        image_path = self.image_root / example[self.cfg.image_field]
        with Image.open(image_path) as img:
            image = img.copy()
        caption = example[self.cfg.caption_field]
        key = example.get(self.cfg.id_field) or image_path.stem

        return build_cache_sample(
            image,
            key,
            self.buckets,
            self.caption_lookup,
            self.alt_caption_lookup,
            caption=caption,
            alt_caption=caption,
        )


def build_densefusion_dataset(
    cfg: CacheDenseFusionDatasetConfig,
    buckets: Iterable[tuple[int, int]],
    caption_lookup: TextLookupBase | None,
    alt_caption_lookup: TextLookupBase | None,
):
    return DenseFusionCacheDataset(cfg, buckets, caption_lookup, alt_caption_lookup)


__all__ = ["build_densefusion_dataset"]
