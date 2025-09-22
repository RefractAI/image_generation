"""Hugging Face Parquet source for cache generation.

Loads one or more Parquet shards with image and label columns and emits
preprocessed samples for the cache pipeline. If the label column contains
class names in the feature metadata (ClassLabel.names), they are used as
caption text; otherwise, the numeric label index is used.
"""

from __future__ import annotations

from typing import Iterable

from datasets import Image as HFImage, load_dataset
from torch.utils.data import Dataset

from config import CacheHFParquetDatasetConfig
from data.text_lookup import TextLookupBase

from .common import build_cache_sample


class HFParquetCacheDataset(Dataset):
    def __init__(
        self,
        cfg: CacheHFParquetDatasetConfig,
        buckets: Iterable[tuple[int, int]],
        caption_lookup: TextLookupBase | None,
        alt_caption_lookup: TextLookupBase | None,
    ) -> None:
        self.cfg = cfg
        data_files = cfg.data_files

        # Load Parquet files directly; list or glob string is supported.
        ds = load_dataset("parquet", data_files=data_files)
        # load_dataset may return a DatasetDict when multiple splits are inferred.
        # We only need a flat Dataset; prefer the first split if a dict is returned.
        if hasattr(ds, "keys") and callable(getattr(ds, "keys", None)):
            # Deterministically pick the first split (e.g., 'train')
            first_key = sorted(ds.keys())[0]
            ds = ds[first_key]

        # Ensure images decode to PIL.Image and return Python types for __getitem__
        if self.cfg.image_field in ds.features and not isinstance(ds.features[self.cfg.image_field], HFImage):
            ds = ds.cast_column(self.cfg.image_field, HFImage(decode=True))
        ds = ds.with_format("python")

        # Prepare a label -> name mapping if available
        self._label_map: dict[int, str] | None = None
        label_feat = ds.features.get(self.cfg.label_field)
        label_names = getattr(label_feat, "names", None)
        if isinstance(label_names, (list, tuple)) and label_names:
            self._label_map = {int(idx): str(name) for idx, name in enumerate(label_names)}

        self.dataset = ds
        self.buckets = buckets
        self.caption_lookup = caption_lookup
        self.alt_caption_lookup = alt_caption_lookup

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.dataset)

    def __getitem__(self, index: int):
        example = self.dataset[index]

        image = example.get(self.cfg.image_field)
        label_value = example.get(self.cfg.label_field)

        label_index_int: int | None = None
        label_index_str = ""
        if label_value is not None:
            try:
                label_index_int = int(label_value)
            except (TypeError, ValueError):
                label_index_str = str(label_value)
            else:
                label_index_str = str(label_index_int)

        # Prefer class name from metadata; fall back to numeric label string
        if self._label_map is not None and label_index_int is not None:
            caption = self._label_map.get(label_index_int, "")
        else:
            caption = label_index_str

        key = index
        return build_cache_sample(
            image,
            key,
            self.buckets,
            self.caption_lookup,
            self.alt_caption_lookup,
            caption=caption,
            alt_caption=label_index_str,
        )


def build_hf_parquet_dataset(
    cfg: CacheHFParquetDatasetConfig,
    buckets: Iterable[tuple[int, int]],
    caption_lookup: TextLookupBase | None,
    alt_caption_lookup: TextLookupBase | None,
):
    dataset = HFParquetCacheDataset(cfg, buckets, caption_lookup, alt_caption_lookup)
    return dataset


__all__ = ["build_hf_parquet_dataset"]
