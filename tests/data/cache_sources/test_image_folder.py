from pathlib import Path
from PIL import Image

from data.cache_sources.image_folder import build_image_folder_dataset
from config import CacheImageFolderConfig


def test_image_folder_reads_sidecar_caption(tmp_path: Path):
    # Create image and sidecar caption
    root = tmp_path / "imgs"
    root.mkdir()
    img_path = root / "sample.jpg"
    Image.new("RGB", (32, 48), color=(123, 222, 64)).save(img_path)
    (root / "sample.txt").write_text("sidecar caption", encoding="utf-8")

    cfg = CacheImageFolderConfig(root=str(root), extensions=[".jpg"], recursive=False)
    dataset = build_image_folder_dataset(cfg, buckets=[(256, 256)], caption_lookup=None, alt_caption_lookup=None)

    tensor, captions, key, caption, orig_res, target_res = dataset[0]
    assert captions == "sidecar caption"
    assert isinstance(key, str)
    assert orig_res == (32, 48)
    assert target_res == (256, 256)

