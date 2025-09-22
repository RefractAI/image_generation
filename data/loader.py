import multiprocessing as mp

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from .bucketing import BucketBatchSampler

def load_dataset(cfg):
    match cfg.type:
        case "huggingface_disk":
            return load_from_disk(cfg.path).with_format("torch")
        case _:
            raise ValueError(f"Unsupported dataset type '{cfg.type}'.")

def build_loader(dataset, cfg, bucket_list):
    num_workers = min(cfg.max_workers, mp.cpu_count())    
    dataset_id = f"{cfg.type}:{cfg.path}" if getattr(cfg, "path", None) else cfg.type
    if cfg.multires:
        sampler = BucketBatchSampler(
            dataset,
            bucket_list,
            cfg.batch_size,
            drop_last=True,
            dataset_id=dataset_id,
        )
        loader = DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers if torch.cuda.is_available() else 0, pin_memory=torch.cuda.is_available(), persistent_workers=torch.cuda.is_available())
    else:
        sampler = None
        loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=num_workers if torch.cuda.is_available() else 0, pin_memory=torch.cuda.is_available(), persistent_workers=torch.cuda.is_available())
    return loader, sampler
