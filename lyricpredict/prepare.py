from __future__ import annotations

import argparse
from dataclasses import asdict

from .config import load_config
from .importer import prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean raw lyric files into train/valid text files.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--source-dir", default=None, help="Optional lyric source directory override.")
    args = parser.parse_args()
    config = load_config(args.config)
    raw_dir = config.paths.raw_dir if args.source_dir is None else config.paths.raw_dir.parent.parent / args.source_dir
    stats = prepare_dataset(raw_dir, config.paths.processed_dir, config.training.validation_ratio)
    print(asdict(stats))


if __name__ == "__main__":
    main()
