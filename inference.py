import argparse

from config import load_inference_config
from inference.engine import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_inference_config(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
