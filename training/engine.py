import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

import torch
from accelerate import Accelerator
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from transformers import get_cosine_schedule_with_warmup
from tqdm.auto import tqdm

from data.bucketing import get_sample_batch
from config import TrainConfig, OptimizerConfig
from data.loader import load_dataset, build_loader
from data.aspect_ratios import bucket_spec
from models.model import build_model
from models.text import build_text_stack, encode_text
from models.vae import build_vae
from training.checkpoint import save_checkpoint, load_checkpoint
from training.loop_context import TrainingLoopContext
from training.sample_step import default_sample_step
from training.train_step import default_train_step
from training.losses import LossManager


def _create_optimizer(model: torch.nn.Module, optimizer_cfg: OptimizerConfig) -> optim.Optimizer:
    """Construct an optimizer based on the configured name."""

    name = optimizer_cfg.name.lower()
    params = model.parameters()

    match name:
        case "adam":
            return optim.Adam(params, lr=optimizer_cfg.lr)
        case "adamw":
            return optim.AdamW(params, lr=optimizer_cfg.lr)
        case "adamw_fused":
            return optim.AdamW(params, lr=optimizer_cfg.lr, fused=True) if torch.cuda.is_available() else optim.AdamW(params, lr=optimizer_cfg.lr)
        case _:
            raise ValueError(f"Unsupported optimizer '{optimizer_cfg.name}'.")


def train(cfg: TrainConfig, *, config_path: str | None = None):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    accelerator = Accelerator(mixed_precision=cfg.training.mixed_precision if torch.cuda.is_available() else "no")
    dataset = load_dataset(cfg.dataset)
    bucket_list, _, _ = bucket_spec(cfg.dataset.image_size)
    loader, sampler = build_loader(dataset, cfg.dataset, bucket_list)
    tokenizer, text_encoder = build_text_stack(cfg, accelerator.device.type)
    vae_device = accelerator.device if accelerator.num_processes == 1 else torch.device("cuda", 0)
    vae = build_vae(cfg.vae.class_name, cfg.vae.path, vae_device)
    model = build_model(cfg.model, cfg.text_encoder)
    optimizer = _create_optimizer(model, cfg.optimizer)
    scheduler = get_cosine_schedule_with_warmup(optimizer, cfg.optimizer.warmup_steps, cfg.training.epochs * len(loader))
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    if text_encoder is not None:
        text_encoder = accelerator.prepare_model(text_encoder, evaluation_mode=True)
    loader = accelerator.prepare_data_loader(loader)
    if text_encoder is not None:
        for parameter in text_encoder.parameters():
            parameter.requires_grad = False
    dtype = next(model.parameters()).dtype
    encode = lambda captions: encode_text(cfg, tokenizer, text_encoder, accelerator.device, cfg.text_encoder.max_length, dtype, captions)
    uncond = encode([cfg.training.unconditional_caption])
    checkpoints_dir = Path(cfg.logging.checkpoints_dir)
    samples_dir = Path(cfg.logging.samples_dir)
    runs_dir = Path(cfg.logging.runs_dir)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    config_name = Path(config_path).stem if config_path else "run"
    run_log_dir = runs_dir / f"{config_name}-{run_timestamp}"
    if accelerator.is_main_process:
        run_log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_log_dir)) if accelerator.is_main_process else None
    global_step = 0
    if cfg.training.resume:
        global_step, _ = load_checkpoint(accelerator, model, optimizer, scheduler, checkpoints_dir)
    loss_manager = LossManager(cfg.losses, cfg.training.log_window)
    match cfg.loop.train_step:
        case "default":
            train_step_fn = default_train_step
        case _:
            raise ValueError(f"Unsupported train loop '{cfg.loop.train_step}'.")

    match cfg.loop.sample_step:
        case "default":
            sample_step_fn = default_sample_step
        case _:
            raise ValueError(f"Unsupported sample loop '{cfg.loop.sample_step}'.")

    loop_context = TrainingLoopContext(
        accelerator=accelerator,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        encode=encode,
        uncond=uncond,
        cfg=cfg,
        loss_manager=loss_manager,
        vae=vae,
        samples_dir=samples_dir,
        writer=writer,
        track_performance=cfg.logging.track_performance,
    )
    (
        loop_context.fixed_images,
        loop_context.fixed_labels,
        loop_context.fixed_noise,
    ) = get_sample_batch(
        dataset,
        loader,
        sampler,
        context=loop_context,
        cfg=cfg,
    )

    if accelerator.is_main_process:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Labels:", loop_context.fixed_labels.original_captions)

    compile_cache_bytes_path = checkpoints_dir / "torch_compile_cache.bin"

    if cfg.model.compile and torch.cuda.is_available() and compile_cache_bytes_path.exists():
        artifact_bytes = compile_cache_bytes_path.read_bytes()
        torch.compiler.load_cache_artifacts(artifact_bytes)
        if accelerator.is_main_process:
            print(f"Loaded torch.compile cache artifacts from {compile_cache_bytes_path}")

    for epoch in range(cfg.training.epochs):
        if sampler and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        progress = tqdm(loader, leave=False, desc=f"epoch {epoch}") if accelerator.is_main_process else loader
        data_timer = perf_counter()
        for batch in progress:
            if loop_context.track_performance:
                loop_context.performance_timings.clear()
                loop_context.performance_timings["dataset_sample"] = perf_counter() - data_timer
            loop_context.global_step = global_step
            result = train_step_fn(batch=batch, context=loop_context, global_step=global_step)
            if not isinstance(result, dict) or "loss" not in result:
                raise ValueError("Train step function must return a dict containing a 'loss' tensor.")
            loss = result["loss"]
            if not torch.is_tensor(loss):
                raise TypeError("Train step function must return a tensor for 'loss'.")
            if accelerator.is_main_process:
                should_update = (global_step % 10 == 0)
                if should_update:
                    loss_manager.update(result)
                    averages = loss_manager.averages()
                    lr = optimizer.param_groups[0]["lr"]
                    postfix = {name: f"{value:.4f}" for name, value in averages.items()}
                    postfix["lr"] = f"{lr:.2e}"
                    progress.set_postfix(postfix)
                    if writer:
                        for name, value in averages.items():
                            writer.add_scalar(f"train/{name}", float(value), global_step)
                        # Log LR sparsely to reduce sync
                        writer.add_scalar("train/learning_rate", lr, global_step)
                        if loop_context.track_performance:
                            for name, value in loop_context.performance_timings.items():
                                writer.add_scalar(f"perf/{name}", float(value), global_step)
            
            if accelerator.is_main_process and global_step != 0 and global_step % cfg.training.save_every == 0:
                save_checkpoint(accelerator, model, optimizer, scheduler, global_step, checkpoints_dir)
                
            if accelerator.is_main_process and global_step % cfg.sampling.interval == 0:
                print("Sampling")
                sample_step_fn(context=loop_context, global_step=global_step)
                print("Finished sampling")

            if cfg.model.compile and torch.cuda.is_available() and not compile_cache_bytes_path.exists() and global_step == 50:
                artifacts = torch.compiler.save_cache_artifacts()
                artifact_bytes, _ = artifacts
                compile_cache_bytes_path.write_bytes(artifact_bytes)
                print(
                    f"Saved torch.compile cache artifacts to {compile_cache_bytes_path} at step {global_step}"
                )

            global_step += 1
            if loop_context.track_performance:
                data_timer = perf_counter()
