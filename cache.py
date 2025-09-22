import argparse
import gc
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torchvision
from datasets import Dataset, concatenate_datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import multiprocessing as mp

from config import CacheConfig, load_cache_config
from data.aspect_ratios import bucket_spec
from data.cache_sources import build_cache_dataset
from data.text_lookup import create_text_lookup, TextLookupBase
from models.vae import BaseVAE, build_vae


torch.multiprocessing.set_sharing_strategy('file_system')


class AspectRatioBatcher:
    def __init__(self, buckets, bucket_size=16):
        self.bucket_size = bucket_size
        self.bucket_keys = list(buckets)
        self.buckets = {bucket: [] for bucket in self.bucket_keys}
        
    def add_sample(self, sample):
        image, _, _, _, _, _ = sample
        _, H, W = image.shape
        bucket = (W, H)
        
        if bucket in self.buckets:
            self.buckets[bucket].append(sample)
            
            # Return batch if bucket is full
            if len(self.buckets[bucket]) >= self.bucket_size:
                batch = self.buckets[bucket][:self.bucket_size]
                self.buckets[bucket] = self.buckets[bucket][self.bucket_size:]
                return batch
        return None
    
    def get_remaining_batches(self):
        batches = []
        for bucket in self.buckets:
            if self.buckets[bucket]:
                batches.append(self.buckets[bucket])
                self.buckets[bucket] = []
        return batches

def collate_fn(batch):
    # Simple collate for individual samples
    return batch[0] if len(batch) == 1 else batch

def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def get_next_cache_number(cache_dir: Path) -> int:
    existing_caches = [path for path in cache_dir.glob("dataset_part_*") if path.is_dir()]
    if not existing_caches:
        return 0

    numbers: list[int] = []
    for path in existing_caches:
        try:
            numbers.append(int(path.name.split('_')[-1]))
        except ValueError:
            continue

    return max(numbers) + 1 if numbers else 0

def cache_dataset(
    dataloader: DataLoader,
    device: torch.device,
    autoencoder: BaseVAE,
    cfg: CacheConfig,
    buckets: Iterable[tuple[int, int]],
    text_lookup: TextLookupBase | None,
):
    cache_dir = Path(cfg.output.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    batcher = AspectRatioBatcher(buckets=buckets, bucket_size=cfg.processing.bucket_size)
    save_threshold = cfg.processing.save_threshold
    min_caption_length = cfg.processing.min_caption_length

    next_cache = get_next_cache_number(cache_dir)

    all_latents: list[np.ndarray] = []
    all_keys: list[str] = []
    all_captions: list[str] = []
    all_alt_captions: list[str] = []
    all_original_resolutions: list[tuple[int, int]] = []
    all_target_resolutions: list[tuple[int, int]] = []
    processed_count = 0

    def process_batch(batch_samples):
        if not batch_samples:
            return [], [], [], [], [], []

        images, captions, keys, alt_captions, original_resolutions, target_resolutions = zip(*batch_samples)
        image_batch = torch.stack(images)

        with torch.no_grad():
            latents = autoencoder.encode(image_batch.to(device))
            latents_list = latents.cpu().numpy()

        return (
            latents_list,
            list(captions),
            list(keys),
            list(alt_captions),
            list(original_resolutions),
            list(target_resolutions),
        )

    def save_accumulated_data():
        nonlocal next_cache, all_latents, all_keys, all_captions, all_alt_captions, all_original_resolutions, all_target_resolutions, processed_count

        if not all_latents:
            return

        current_dataset = Dataset.from_dict(
            {
                'latents': all_latents,
                'keys': all_keys,
                'captions': all_captions,
                'alt_captions': all_alt_captions,
                'original_resolution': all_original_resolutions,
                'target_resolution': all_target_resolutions,
            }
        )
        save_path = cache_dir / f"dataset_part_{next_cache}"
        print(f"\nSaving {len(all_latents)} samples to {save_path}")
        current_dataset.save_to_disk(str(save_path))
        next_cache += 1

        all_latents = []
        all_keys = []
        all_captions = []
        all_alt_captions = []
        all_original_resolutions = []
        all_target_resolutions = []
        processed_count = 0
        gc.collect()

    try:
        pbar = tqdm(dataloader)
        for idx, sample in enumerate(pbar):
            try:
                _, caption, _, _, _, _ = sample

                if len(caption) <= min_caption_length:
                    continue

                batch = batcher.add_sample(sample)

                if batch is not None:
                    (
                        latents_list,
                        captions_list,
                        keys_list,
                        alt_captions_list,
                        original_resolutions_list,
                        target_resolutions_list,
                    ) = process_batch(batch)

                    all_latents.extend(latents_list)
                    all_keys.extend(keys_list)
                    all_captions.extend(captions_list)
                    all_alt_captions.extend(alt_captions_list)
                    all_original_resolutions.extend(original_resolutions_list)
                    all_target_resolutions.extend(target_resolutions_list)
                    processed_count += len(latents_list)

                    if processed_count >= save_threshold:
                        save_accumulated_data()

                bucket_sizes = {
                    f"{b[0]}x{b[1]}": len(samples)
                    for b, samples in batcher.buckets.items()
                    if samples
                }
                pbar.set_description(f"Sample {idx}, buckets: {bucket_sizes}")

            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                continue

        print("\nProcessing remaining samples in buckets...")
        remaining_batches = batcher.get_remaining_batches()
        for batch in tqdm(remaining_batches):
            if not batch:
                continue

            (
                latents_list,
                captions_list,
                keys_list,
                alt_captions_list,
                original_resolutions_list,
                target_resolutions_list,
            ) = process_batch(batch)

            all_latents.extend(latents_list)
            all_keys.extend(keys_list)
            all_captions.extend(captions_list)
            all_alt_captions.extend(alt_captions_list)
            all_original_resolutions.extend(original_resolutions_list)
            all_target_resolutions.extend(target_resolutions_list)

        save_accumulated_data()
    except Exception as e:
        print(f"Fatal error during caching loop: {e}")


def run_caching(cfg: CacheConfig):
    aspect_ratio = cfg.aspect_ratio_bucket
    try:
        bucket_list, _, single_sample_res = bucket_spec(aspect_ratio)
    except KeyError as err:
        raise ValueError(f"Unsupported aspect ratio bucket size: {aspect_ratio}") from err

    if cfg.processing.multires:
        buckets = list(bucket_list)
    else:
        square_res = tuple(int(v) for v in single_sample_res)
        buckets = [square_res]
    caption_lookup: TextLookupBase | None = None
    alt_lookup: TextLookupBase | None = None
    if getattr(cfg, "caption_text_lookup", None) and cfg.caption_text_lookup.type and cfg.caption_text_lookup.path:
        caption_lookup = create_text_lookup(cfg.caption_text_lookup)
    if getattr(cfg, "alt_text_lookup", None) and cfg.alt_text_lookup.type and cfg.alt_text_lookup.path:
        alt_lookup = create_text_lookup(cfg.alt_text_lookup)

    dataset_result = build_cache_dataset(cfg, buckets, caption_lookup, alt_lookup)

    n_workers = min(cfg.dataloader.max_workers, mp.cpu_count())
    if cfg.dataset.type == 'webdataset':
        n_workers = min(n_workers, cfg.dataset.webdataset.max_index - cfg.dataset.webdataset.min_index)

    print(f"Using {n_workers} worker(s) for dataset type '{cfg.dataset.type}'")

    device = _select_device()
    autoencoder = build_vae(cfg.vae.class_name, cfg.vae.path, device)

    dataloader = DataLoader(
        dataset_result.dataset,
        batch_size=1,
        num_workers=n_workers,
        collate_fn=collate_fn,
    )
    cache_dataset(dataloader, device, autoencoder, cfg, buckets, caption_lookup)

def concatenate_cached_datasets(cache_dir: Path):
    part_idx = 0
    all_datasets = []

    while True:
        dataset_path = cache_dir / f"dataset_part_{part_idx}"
        if not dataset_path.exists():
            break
        ds = Dataset.load_from_disk(str(dataset_path))
        all_datasets.append(ds)
        part_idx += 1

    if not all_datasets:
        raise ValueError(f"No dataset parts found in cache directory {cache_dir}")

    combined_dataset = concatenate_datasets(all_datasets)
    combined_dataset = combined_dataset.with_format(type="torch", columns=["latents"], output_all_columns=True)

    combined_path = cache_dir / "combined_dataset"
    combined_dataset.save_to_disk(str(combined_path))

    print(f"Combined {part_idx} dataset parts into sharded dataset")
    print(f"Total samples: {len(combined_dataset)}")

    return combined_dataset


def test_cached_dataset(cache_dir: Path, cfg: CacheConfig, sample_index: int = 0):
    dataset_path = cache_dir / "combined_dataset"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Combined dataset not found at {dataset_path}")

    ds = Dataset.load_from_disk(str(dataset_path))
    if len(ds) == 0:
        raise ValueError("Combined dataset is empty")

    key = min(sample_index, len(ds) - 1)
    sample = ds[key]

    print("\nDataset summary:")
    print(f"Number of samples: {len(ds)}")
    print(f"Inspecting sample index: {key}")
    print(f"Latents type: {type(sample['latents'])}")
    print(f"Sample key: {sample['keys']}")
    print(f"Sample caption: {sample['captions']}")
    print(f"Sample alt caption: {sample['alt_captions']}")
    print(f"Sample captions length: {len(sample['captions']) if isinstance(sample['captions'], str) else 'n/a'}")
    print(f"Sample alt_caption length: {len(sample['alt_captions']) if isinstance(sample['alt_captions'], str) else 'n/a'}")
    print(f"Sample original resolution: {sample['original_resolution']}")
    print(f"Sample target resolution: {sample['target_resolution']}")

    latents = sample['latents']
    if not isinstance(latents, torch.Tensor):
        latents_tensor = torch.as_tensor(latents)
    else:
        latents_tensor = latents
    latents_tensor = latents_tensor.unsqueeze(0)

    device = _select_device()
    autoencoder = build_vae(cfg.vae.class_name, cfg.vae.path, device)
    decoded_image = autoencoder.decode(latents_tensor.to(device))
    decoded_image = decoded_image.clamp(-1.0, 1.0)
    decoded_image = (decoded_image + 1.0) * 0.5

    output_path = "decoded_image.png"
    torchvision.utils.save_image(decoded_image.cpu(), str(output_path))
    print(f"Saved decoded sample to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Cache dataset shards using YAML configuration")
    parser.add_argument("--config", default="configs/cache.yaml")
    args = parser.parse_args()

    cfg = load_cache_config(args.config)
    cache_dir = Path(cfg.output.cache_dir)

    run_caching(cfg)

    concatenate_cached_datasets(cache_dir)

    test_cached_dataset(cache_dir, cfg)


if __name__ == "__main__":
    main()
