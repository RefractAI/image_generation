import argparse

from config import load_train_config
from training.engine import train
import os
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_train_config(args.config)
    train(cfg, config_path=args.config)


if __name__ == "__main__":
    main()
