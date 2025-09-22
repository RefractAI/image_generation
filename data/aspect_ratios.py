"""Shared aspect ratio bucket definitions."""

from __future__ import annotations

# Common bucket layouts (width, height) pairs kept centralized so they can be
# reused across data preparation utilities.
aspect_ratio_buckets_1024 = (
    (512, 2048),
    (576, 1664),
    (640, 1536),
    (704, 1408),
    (768, 1280),
    (832, 1216),
    (896, 1152),
    (960, 1088),
    (1024, 1024),
    (1088, 960),
    (1152, 896),
    (1216, 832),
    (1280, 768),
    (1408, 704),
    (1536, 640),
    (1664, 576),
    (2048, 512),
)

aspect_ratio_buckets_512 = (
    (256, 1024),
    (288, 832),
    (320, 768),
    (352, 704),
    (384, 640),
    (416, 608),
    (448, 576),
    (480, 544),
    (512, 512),
    (544, 480),
    (576, 448),
    (608, 416),
    (640, 384),
    (704, 352),
    (768, 320),
    (832, 288),
    (1024, 256),
)

aspect_ratio_buckets_256 = (
    (128, 512),
    (160, 384),
    (192, 320),
    (224, 288),
    (256, 256),
    (288, 224),
    (320, 192),
    (384, 160),
    (512, 128),
)

def find_closest_aspect_ratio(width, height, buckets):
    aspect_ratio = width / height
    closest_bucket = min(
        buckets,
        key=lambda bucket: abs(bucket[0] / bucket[1] - aspect_ratio)
    )
    return closest_bucket

__all__ = [
    "aspect_ratio_buckets_1024",
    "aspect_ratio_buckets_512",
    "aspect_ratio_buckets_256",
    "bucket_spec",
    "find_closest_aspect_ratio"
]


def bucket_spec(image_size: int):
    table = {
        256: (aspect_ratio_buckets_256, (224, 288), (256, 256)),
        512: (aspect_ratio_buckets_512, (448, 576), (512, 512)),
        1024: (aspect_ratio_buckets_1024, (896, 1152), (1024, 1024)),
    }
    try:
        return table[image_size]
    except KeyError as err:
        raise KeyError(f"Unsupported image size {image_size} for bucket_spec.") from err
