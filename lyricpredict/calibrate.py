from __future__ import annotations

import argparse
import statistics

from .confidence import ConfidenceSettings, save_confidence_settings
from .config import load_config
from .generation import LyricGenerator


def percentile(values: list[float], pct: int) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    index = round((len(values) - 1) * pct / 100)
    return values[index]


def context_from_line(line: str) -> str:
    line = line.strip()
    if len(line) <= 4:
        return line
    return line[: max(2, len(line) // 2)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate confidence threshold from validation lyric contexts.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--max-samples", type=int, default=64)
    args = parser.parse_args()
    config = load_config(args.config)
    valid_path = config.paths.processed_dir / "valid.txt"
    if not valid_path.exists():
        raise FileNotFoundError(f"Validation file not found: {valid_path}")
    lines = [line.strip() for line in valid_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError("Validation file is empty.")

    generator = LyricGenerator(config)
    scores: list[float] = []
    for line in lines[: args.max_samples]:
        prediction = generator.predict(context_from_line(line))
        if prediction.confidence > 0:
            scores.append(prediction.confidence)

    threshold = percentile(scores, config.confidence.calibration_percentile) if scores else config.confidence.threshold
    settings = ConfidenceSettings(
        threshold=threshold,
        min_token_probability=config.confidence.min_token_probability,
        max_repeat_ratio=config.confidence.max_repeat_ratio,
    )
    save_confidence_settings(config.paths.model_dir, settings, scores)
    mean = statistics.mean(scores) if scores else 0.0
    print({"threshold": threshold, "samples": len(scores), "mean": mean})


if __name__ == "__main__":
    main()
