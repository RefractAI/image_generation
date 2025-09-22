from pathlib import Path
from PIL import Image

import pyarrow as pa
import pyarrow.parquet as pq

from data.cache_sources.hf_parquet import build_hf_parquet_dataset
import data.cache_sources.hf_parquet as hfmod
from config import CacheHFParquetDatasetConfig


def _write_parquet_with_paths(path: Path, image_paths: list[str], labels: list[int]) -> None:
    table = pa.table({
        "image": pa.array(image_paths, type=pa.string()),
        "label": pa.array(labels, type=pa.int32()),
    })
    pq.write_table(table, path)


def test_hf_parquet_dataset_reads_images_and_labels(tmp_path: Path, monkeypatch):
    # Isolate HF datasets cache to avoid permission issues
    hf_cache = tmp_path / "hf-cache"
    monkeypatch.setenv("HF_DATASETS_CACHE", str(hf_cache))
    monkeypatch.setenv("HF_HOME", str(hf_cache))
    # Ensure datasets picks up env by forcing re-import
    import sys
    sys.modules.pop("datasets", None)
    # Create two images in-memory for a dummy dataset
    img1 = Image.new("RGB", (16, 24), color=(1, 2, 3))
    img2 = Image.new("RGB", (24, 16), color=(4, 5, 6))

    class DummyDS:
        def __init__(self, items):
            self._items = items
            self.features = {}
        def with_format(self, fmt):
            return self
        def cast_column(self, name, t):
            return self
        def __len__(self):
            return len(self._items)
        def __getitem__(self, i):
            return self._items[i]

    dummy = DummyDS([
        {"image": img1, "label": 7},
        {"image": img2, "label": 8},
    ])
    monkeypatch.setattr(hfmod, "load_dataset", lambda *a, **k: dummy)

    cfg = CacheHFParquetDatasetConfig(data_files="ignored", image_field="image", label_field="label")
    ds = build_hf_parquet_dataset(cfg, buckets=[(256, 256)], caption_lookup=None, alt_caption_lookup=None)

    sample0 = ds[0]
    t0, caps0, key0, cap0, orig0, tgt0 = sample0
    assert caps0 == "7"  # falls back to numeric label string
    assert cap0 == "7"  # alt caption uses numeric class index by default
    assert isinstance(key0, str)
    assert orig0 == (16, 24)
    assert tgt0 == (256, 256)


def test_hf_parquet_alt_caption_preserves_label_index_with_names(monkeypatch):
    class DummyDS:
        def __init__(self):
            self._items = [{"image": Image.new("RGB", (32, 32)), "label": 1}]

            class LabelFeature:
                names = ["zero", "one"]

            self.features = {"label": LabelFeature()}

        def with_format(self, fmt):
            return self

        def cast_column(self, name, t):
            return self

        def __len__(self):
            return len(self._items)

        def __getitem__(self, idx):
            return self._items[idx]

    monkeypatch.setattr(hfmod, "load_dataset", lambda *a, **k: DummyDS())

    cfg = CacheHFParquetDatasetConfig(data_files="ignored", image_field="image", label_field="label")
    ds = build_hf_parquet_dataset(cfg, buckets=[(256, 256)], caption_lookup=None, alt_caption_lookup=None)

    _, captions, _, alt_caption, _, _ = ds[0]
    assert captions == "one"  # class names courtesy of metadata
    assert alt_caption == "1"  # alt caption should retain the class index
