import io
import tarfile
from pathlib import Path
from PIL import Image
from PIL import features
import pytest

from data.cache_sources.webdataset import build_webdataset_dataset
from config import CacheShardConfig


def _make_tar_with_webp(path: Path, member_name: str = "sample.webp") -> None:
    img = Image.new("RGB", (40, 40), color=(10, 20, 30))
    buf = io.BytesIO()
    img.save(buf, format="WEBP")
    data = buf.getvalue()
    with tarfile.open(path, mode="w") as tf:
        info = tarfile.TarInfo(name=member_name)
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))


def test_webdataset_builds_and_emits_samples(tmp_path: Path):
    if not features.check("webp"):
        pytest.skip("Pillow WEBP support not available")
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    tar_path = shard_dir / "0000.tar"
    _make_tar_with_webp(tar_path)

    cfg = CacheShardConfig(directory=str(shard_dir), pattern="%04d.tar", min_index=0, max_index=0)
    ds = build_webdataset_dataset(cfg, buckets=[(256, 256)], caption_lookup=None, alt_caption_lookup=None)

    sample = next(iter(ds))
    tensor, captions, key, caption, orig_res, target_res = sample
    assert tensor.shape[0] == 3
    assert captions == ""
    assert caption == ""
    assert isinstance(key, str)
    assert orig_res == (40, 40)
    assert target_res == (256, 256)
