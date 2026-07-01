"""
benchmark_worker.py

Entry point for a single benchmark worker.

Responsibilities:
1. Read the experiment configuration.
2. Assign this worker's rank.
3. Launch distributed training.
"""

import argparse
import json

from dist_launcher import launch_distributed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark worker"
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to experiment config.json",
    )

    parser.add_argument(
        "--rank",
        type=int,
        required=True,
        help="Worker rank",
    )

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def main():

    args = parse_args()

    config = load_config(args.config)

    config["rank"] = args.rank
    print(f"Worker started with rank {config['rank']}")

    launch_distributed(config)


if __name__ == "__main__":
    main()