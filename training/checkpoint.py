from pathlib import Path

import torch


def save_checkpoint(accelerator, model, optimizer, scheduler, step, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"checkpoint_{step}.pth"
    print(f"Checkpoint saved to {path}")
    accelerator.save({"model": accelerator.unwrap_model(model).state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "step": step}, path)
    return path


def load_checkpoint(accelerator, model, optimizer, scheduler, directory):
    directory = Path(directory)
    ckpts = sorted(directory.glob("checkpoint_*.pth"), key=lambda x: int(x.stem.split("_")[1]))
    if len(ckpts) > 0:
        path = ckpts[-1]
        print(f"Checkpoint loaded from {path}")
        state = torch.load(path, map_location="cpu")
        accelerator.unwrap_model(model).load_state_dict({k.replace("_orig_mod.", ""): v for k, v in state["model"].items()}, strict=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        return state["step"]+1, path
    else:
        print("No checkpoints found, training from scratch")
        return 0, ""
