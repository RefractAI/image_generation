"""Shared helpers for cache dataset sources."""

from __future__ import annotations

from typing import Iterable, Tuple

from PIL import Image
import torchvision.transforms as T

from data.aspect_ratios import find_closest_aspect_ratio
from data.text_lookup import TextLookupBase

def build_cache_sample(
    image,
    key: object,
    buckets: Iterable[Tuple[int, int]],
    caption_lookup: TextLookupBase | None,
    alt_caption_lookup: TextLookupBase | None,
    caption: str = "",
    alt_caption: str = "",
):
    """Transform an image into the tuple expected by the cache pipeline."""

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    image = image.convert("RGB")
    width, height = image.size
    target_width, target_height = find_closest_aspect_ratio(width, height, buckets)
    target_size = max(target_width, target_height)

    transform = T.Compose(
        [
            T.Resize(target_size),
            T.CenterCrop((target_height, target_width)),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    tensor = transform(image)

    captions_text = caption
    alt_caption_text = alt_caption
    if caption_lookup is not None:
        lookup_captions, _ = caption_lookup.lookup(key)
        if lookup_captions:
            captions_text = lookup_captions
    if alt_caption_lookup is not None:
        _, lookup_caption = alt_caption_lookup.lookup(key)
        if lookup_caption:
            alt_caption_text = lookup_caption

    return tensor, captions_text, str(key), alt_caption_text, (width, height), (target_width, target_height)


__all__ = ["build_cache_sample"]
