"""Image folder dataset source for cache generation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image
from torch.utils.data import Dataset

from config import CacheImageFolderConfig
from data.text_lookup import TextLookupBase

from .common import build_cache_sample


_DEFAULT_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _gather_files(root: Path, recursive: bool, extensions: set[str]) -> list[Path]:
    if recursive:
        iterator: Iterable[Path] = root.rglob("*")
    else:
        iterator = root.glob("*")
    return [path for path in iterator if path.is_file() and path.suffix.lower() in extensions]


class ImageFolderCacheDataset(Dataset):
    def __init__(
        self,
        cfg: CacheImageFolderConfig,
        buckets,
        caption_lookup: TextLookupBase | None,
        alt_caption_lookup: TextLookupBase | None,
    ) -> None:
        root = Path(cfg.root).expanduser()
        if not root.exists():
            raise FileNotFoundError(f"Image folder root '{root}' does not exist")
        extensions = {ext.lower() for ext in (cfg.extensions or _DEFAULT_EXTENSIONS)}
        self.files = sorted(_gather_files(root, cfg.recursive, extensions))
        if not self.files:
            raise ValueError(f"No image files found in '{root}' with extensions {sorted(extensions)}")
        self.root = root
        self.buckets = buckets
        self.caption_lookup = caption_lookup
        self.alt_caption_lookup = alt_caption_lookup

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.files)

    def __getitem__(self, index: int):
        path = self.files[index]
        with Image.open(path) as img:
            image = img.convert("RGB")

        relative_path = path.relative_to(self.root)
        key = str(relative_path)
        # Default caption: read from sibling .txt file if present; otherwise folder name or stem
        txt_path = path.with_suffix(".txt")
        if txt_path.exists():
            try:
                caption = txt_path.read_text(encoding="utf-8").strip()
            except Exception:
                caption = ""
        else:
            caption = relative_path.parent.name if relative_path.parent != Path(".") else path.stem

        return build_cache_sample(
            image,
            key,
            self.buckets,
            self.caption_lookup,
            self.alt_caption_lookup,
            caption=caption,
        )


def build_image_folder_dataset(
    cfg: CacheImageFolderConfig,
    buckets,
    caption_lookup: TextLookupBase | None,
    alt_caption_lookup: TextLookupBase | None,
):
    dataset = ImageFolderCacheDataset(cfg, buckets, caption_lookup, alt_caption_lookup)
    return dataset


__all__ = ["build_image_folder_dataset"]
