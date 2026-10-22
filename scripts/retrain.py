"""Reproduce a recorded TrainConfig into a fresh run directory using uv."""

import argparse
import json
from snakerl.training import TrainConfig, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--init-checkpoint", help="Replace the recorded local checkpoint path")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as stream:
        values = json.load(stream)
    values["output_dir"] = args.out
    if args.init_checkpoint:
        values["init_checkpoint"] = args.init_checkpoint
    print(train(TrainConfig(**values)))


if __name__ == "__main__":
    main()
