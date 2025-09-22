import sys
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as T
from datasets import Image as HFImage
from datasets import load_dataset

import torch_fidelity

from config import CacheConfig, TrainConfig, load_cache_config, load_train_config
from models.model import build_model
from models.text import build_text_stack, encode_text
from models.vae import build_vae
from training.rectified_flow import rectified_flow_sample


NUM_SAMPLES = 50000
BATCH_SIZE = 64
SAMPLING_STEPS = 50
GUIDANCE_SCALE = 2.0
SEED = 0
WORKERS = 4
IMAGE_SIZE = 256
DOWNSAMPLE = 8


class ImageNet256EvalDataset(Dataset):
    def __init__(self, cache_cfg: CacheConfig) -> None:
        data_cfg = cache_cfg.dataset.hf_parquet
        ds = load_dataset("parquet", data_files=data_cfg.data_files)
        ds = next(iter(ds.values()))
        ds = ds.cast_column(data_cfg.image_field, HFImage(decode=True))
        ds = ds.with_format("python")
        self._dataset = ds
        self._image_field = data_cfg.image_field
        self._label_field = data_cfg.label_field
        names = ds.features[self._label_field].names
        self._label_map = {int(idx): str(name) for idx, name in enumerate(names)}
        self._tx = T.Compose([
            T.Resize(IMAGE_SIZE, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(IMAGE_SIZE),
            T.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int):
        sample = self._dataset[index]
        image = sample[self._image_field].convert("RGB")
        img = self._tx(image)
        caption = self._label_map[int(sample[self._label_field])]
        return img, caption


def main():
    if len(sys.argv) != 4:
        raise SystemExit("Usage: python evaluate_fid.py TRAIN_CONFIG CACHE_CONFIG CHECKPOINT")

    train_cfg: TrainConfig = load_train_config(sys.argv[1])
    cache_cfg: CacheConfig = load_cache_config(sys.argv[2])
    checkpoint_path = Path(sys.argv[3])

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run FID evaluation")

    torch.set_float32_matmul_precision("high")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda")

    tokenizer, text_encoder = build_text_stack(train_cfg, device.type)
    text_encoder.to(device)
    vae = build_vae(train_cfg.vae.class_name, train_cfg.vae.path, device)
    model = build_model(train_cfg.model, train_cfg.text_encoder).to(device).eval()

    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in state["model_state_dict"].items()}, strict=True)

    dtype = next(model.parameters()).dtype
    uncond_text = train_cfg.training.unconditional_caption
    null_cond = encode_text(
        tokenizer,
        text_encoder,
        device,
        train_cfg.text_encoder.max_length,
        dtype,
        [uncond_text],
    ) if GUIDANCE_SCALE > 0 else None

    real_ds = ImageNet256EvalDataset(cache_cfg)
    total = min(NUM_SAMPLES, len(real_ds))
    real_loader = DataLoader(
        real_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=WORKERS,
        pin_memory=True,
    )

    tmp_real = tempfile.TemporaryDirectory(prefix="fid_real_")
    tmp_fake = tempfile.TemporaryDirectory(prefix="fid_fake_")
    real_dir = Path(tmp_real.name)
    fake_dir = Path(tmp_fake.name)

    latent_h = IMAGE_SIZE // DOWNSAMPLE
    latent_w = IMAGE_SIZE // DOWNSAMPLE
    channels = train_cfg.model.channels

    processed = 0
    img_idx = 0

    for imgs01, captions in real_loader:
        if processed >= total:
            break

        remaining = total - processed
        if imgs01.shape[0] > remaining:
            imgs01 = imgs01[:remaining]
            captions = captions[:remaining]

        bs = imgs01.shape[0]

        for i in range(bs):
            torchvision.utils.save_image(imgs01[i], str(real_dir / f"{img_idx + i:08d}.png"), normalize=False)

        cond = encode_text(
            tokenizer,
            text_encoder,
            device,
            train_cfg.text_encoder.max_length,
            dtype,
            list(captions),
        )
        noise = torch.randn(bs, channels, latent_h, latent_w, device=device, dtype=dtype)

        with torch.no_grad():
            latents = rectified_flow_sample(
                model,
                noise,
                cond,
                null_cond,
                SAMPLING_STEPS,
                GUIDANCE_SCALE,
            )
            decoded = vae.decode(latents.to(vae.model.device)).to(device)

        fakes01 = (decoded * 0.5 + 0.5).clamp(0, 1)

        for i in range(bs):
            torchvision.utils.save_image(fakes01[i], str(fake_dir / f"{img_idx + i:08d}.png"), normalize=False)

        processed += bs
        img_idx += bs

    metrics = torch_fidelity.calculate_metrics(
        input1=str(real_dir),
        input2=str(fake_dir),
        cuda=True,
        isc=False,
        fid=True,
        kid=False,
        batch_size=BATCH_SIZE,
        verbose=False,
    )

    fid = float(metrics["frechet_inception_distance"])
    print(f"FID (N={processed}): {fid:.4f}")

    tmp_real.cleanup()
    tmp_fake.cleanup()


if __name__ == "__main__":
    main()
