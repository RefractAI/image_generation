from pathlib import Path

import torch
from torchvision.utils import save_image

from config import InferenceConfig
from models.model import build_model
from models.text import build_text_stack, encode_text
from models.vae import build_vae
from training.rectified_flow import rectified_flow_sample


def latest_checkpoint(directory):
    paths = sorted(Path(directory).glob("checkpoint_*.pth"), key=lambda x: int(x.stem.split("_")[1]))
    return paths[-1]


def run(cfg: InferenceConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.inference.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.inference.seed)
    tokenizer, text_encoder = build_text_stack(cfg, device.type)
    if text_encoder is not None:
        text_encoder.to(device)
    vae = build_vae(cfg.vae.class_name, cfg.vae.path, device)
    model = build_model(cfg.model, cfg.text_encoder).to(device)
    model.eval()
    checkpoint = latest_checkpoint(cfg.io.checkpoint_dir)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in state["model_state_dict"].items()}, strict=True)
    dtype = next(model.parameters()).dtype
    cond = encode_text(cfg, tokenizer, text_encoder, device, cfg.text_encoder.max_length, dtype, [cfg.inference.caption])
    null_cond = (
        encode_text(cfg, tokenizer, text_encoder, device, cfg.text_encoder.max_length, dtype, [cfg.inference.negative_caption])
        if cfg.inference.cfg > 0
        else None
    )
    downsample = 8
    latent_h = cfg.inference.height // downsample
    latent_w = cfg.inference.width // downsample
    noise = torch.randn(1, cfg.model.channels, latent_h, latent_w, device=device, dtype=dtype)
    with torch.no_grad():
        latents = rectified_flow_sample(model, noise, cond, null_cond, cfg.inference.steps, cfg.inference.cfg)
        decoded = vae.decode(latents.to(vae.model.device)).to(device)
    image = (decoded * 0.5 + 0.5).clamp(0, 1).cpu()
    output_path = Path(cfg.io.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(image, output_path, normalize=False)
    return output_path
