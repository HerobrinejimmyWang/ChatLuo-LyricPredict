from __future__ import annotations

import argparse
from dataclasses import asdict

from .config import load_config
from .importer import prepare_dataset_from_sources


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean raw lyric files into train/valid text files.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--source-dir",
        action="append",
        default=None,
        help="Optional lyric source directory override. Repeat to combine multiple directories.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    base_dir = config.paths.raw_dir.parent.parent
    source_args = args.source_dir or [str(config.paths.raw_dir)]
    raw_dirs = []
    for source in source_args:
        source_path = config.paths.raw_dir if source == str(config.paths.raw_dir) else base_dir / source
        raw_dirs.append(source_path if source_path.is_absolute() else base_dir / source_path)
    stats = prepare_dataset_from_sources(raw_dirs, config.paths.processed_dir, config.training.validation_ratio)
    print(asdict(stats))


if __name__ == "__main__":
    main()
