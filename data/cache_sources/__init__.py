"""Factory for cache dataset sources."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from config import CacheConfig
from data.text_lookup import TextLookupBase

from .flowers import build_flowers_dataset
from .image_folder import build_image_folder_dataset
from .webdataset import build_webdataset_dataset
from .hf_parquet import build_hf_parquet_dataset
from .densefusion import build_densefusion_dataset


@dataclass
class CacheDatasetBuildResult:
    dataset: Any

def build_cache_dataset(
    cfg: CacheConfig,
    buckets: Iterable[tuple[int, int]],
    caption_lookup: TextLookupBase | None,
    alt_caption_lookup: TextLookupBase | None,
) -> CacheDatasetBuildResult:
    dataset_cfg = cfg.dataset
    dataset_type = dataset_cfg.type.lower()

    match dataset_type:
        case "webdataset":
            if dataset_cfg.webdataset is None:
                raise ValueError("Cache dataset type 'webdataset' requires 'webdataset' settings in config")
            dataset = build_webdataset_dataset(dataset_cfg.webdataset, buckets, caption_lookup, alt_caption_lookup)
        case "flowers":
            if dataset_cfg.flowers is None:
                raise ValueError("Cache dataset type 'flowers' requires 'flowers' settings in config")
            dataset = build_flowers_dataset(dataset_cfg.flowers, buckets, caption_lookup, alt_caption_lookup)
        case "image_folder":
            if dataset_cfg.image_folder is None:
                raise ValueError("Cache dataset type 'image_folder' requires 'image_folder' settings in config")
            dataset = build_image_folder_dataset(dataset_cfg.image_folder, buckets, caption_lookup, alt_caption_lookup)
        case "hf_parquet":
            if dataset_cfg.hf_parquet is None:
                raise ValueError("Cache dataset type 'hf_parquet' requires 'hf_parquet' settings in config")
            dataset = build_hf_parquet_dataset(dataset_cfg.hf_parquet, buckets, caption_lookup, alt_caption_lookup)
        case "densefusion":
            if dataset_cfg.densefusion is None:
                raise ValueError("Cache dataset type 'densefusion' requires 'densefusion' settings in config")
            dataset = build_densefusion_dataset(dataset_cfg.densefusion, buckets, caption_lookup, alt_caption_lookup)
        case _:
            raise ValueError(f"Unsupported cache dataset type '{dataset_cfg.type}'.")

    return CacheDatasetBuildResult(dataset=dataset)


__all__ = ["CacheDatasetBuildResult", "build_cache_dataset"]
