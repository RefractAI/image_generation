"""Dataset preparation utilities for Oxford Flowers."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Tuple

from datasets import DatasetDict, concatenate_datasets, load_dataset as hf_load_dataset
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

_FLOWERS_LABELS_PATH = Path(__file__).resolve().parent / "labels.txt"


def _load_label_map(path: Path = _FLOWERS_LABELS_PATH) -> dict[int, str]:
    if not path.exists():
        return {}
    lines = [line.strip() for line in path.read_text().splitlines()]
    return {idx: line for idx, line in enumerate(lines) if line}


def _closest_bucket(width: int, height: int, buckets: Iterable[Tuple[int, int]]) -> Tuple[int, int]:
    aspect = width / height if height else 1.0
    return min(buckets, key=lambda b: abs((b[0] / b[1]) - aspect))


def prepare_oxford_flowers(cfg, bucket_spec_fn: Callable[[int], tuple]):
    """Return the Oxford Flowers dataset converted to bucketed tensors."""

    dataset = hf_load_dataset(cfg.path)
    if isinstance(dataset, DatasetDict):
        splits = [dataset[name] for name in dataset]
        dataset = concatenate_datasets(splits) if len(splits) > 1 else splits[0]

    bucket_list, _, single_res = bucket_spec_fn(cfg.image_size)
    buckets = [tuple(int(v) for v in res) for res in bucket_list]
    default_res = tuple(int(v) for v in single_res)
    label_map = _load_label_map()
    multires = bool(cfg.multires)
    image_field = cfg.image_field or "image"

    def _transform(example):
        raw = example.get(image_field)
        if raw is None:
            raise KeyError(f"Dataset example missing image field '{image_field}'.")
        if isinstance(raw, list):
            raw = raw[0]
        image = raw
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("RGB")
        width, height = image.size
        if multires:
            target_w, target_h = _closest_bucket(width, height, buckets)
        else:
            target_w, target_h = default_res
        longest = max(target_w, target_h)
        transform = T.Compose(
            [
                T.Resize(longest, interpolation=InterpolationMode.BICUBIC, antialias=True),
                T.CenterCrop((target_h, target_w)),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        latents = transform(image)
        label_idx = int(example["label"]) if "label" in example else -1
        tag = label_map.get(label_idx, str(label_idx)) if label_idx >= 0 else ""
        return {
            "latents": latents,
            "target_resolution": [int(target_w), int(target_h)],
            "label": label_idx,
            "tags": tag,
        }

    remove_columns = list(dataset.column_names)
    dataset = dataset.map(_transform, remove_columns=remove_columns, desc="Preparing Oxford Flowers dataset")
    dataset.set_format(
        type="torch",
        columns=["latents", "target_resolution", "label"],
        output_all_columns=True,
    )
    return dataset


__all__ = ["prepare_oxford_flowers"]
