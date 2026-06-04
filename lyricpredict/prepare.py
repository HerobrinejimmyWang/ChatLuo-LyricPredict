from __future__ import annotations

import argparse
from dataclasses import asdict

from .config import load_config
from .importer import prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean raw lyric files into train/valid text files.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    stats = prepare_dataset(config.paths.raw_dir, config.paths.processed_dir, config.training.validation_ratio)
    print(asdict(stats))


if __name__ == "__main__":
    main()
