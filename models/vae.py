"""VAE implementations and factory selection."""

from __future__ import annotations

from abc import ABC, abstractmethod

from diffusers import AutoencoderKL


class _NoParamsModel:
    """Minimal stub exposing ``parameters`` and ``device`` for compatibility."""

    def __init__(self, device=None):
        self.device = device

    def parameters(self):
        return iter(())


class BaseVAE(ABC):
    """Minimal interface expected by training and inference loops."""

    @abstractmethod
    def encode(self, x):  # pragma: no cover - interface definition
        raise NotImplementedError

    @abstractmethod
    def decode(self, z):  # pragma: no cover - interface definition
        raise NotImplementedError


class PixelSpaceVAE(BaseVAE):
    """No-op VAE that keeps data in pixel space."""

    def __init__(self, path=None, device=None):
        # Parameters are accepted for interface parity with AutoencoderKLVAE.
        self.path = path
        self.device = device
        self.model = _NoParamsModel(device)

    def encode(self, x):
        return x

    def decode(self, z):
        return z

class AutoencoderKLVAE(BaseVAE):
    def __init__(self, path, device):
        self.model = AutoencoderKL.from_pretrained(path).to(device).eval()
        self.scale = self.model.config.scaling_factor

    def encode(self, x):
        return self.model.encode(x).latent_dist.sample() * self.scale

    def decode(self, z):
        return self.model.decode(z / self.scale).sample


def build_vae(class_name: str, path: str, device) -> BaseVAE:
    """Return a VAE instance matching the configured class name."""

    match class_name:
        case "AutoencoderKLVAE":
            return AutoencoderKLVAE(path, device)
        case "PixelSpaceVAE":
            return PixelSpaceVAE(path, device)
        case _:
            raise ValueError(f"Unsupported VAE class '{class_name}'.")
