from __future__ import annotations

import argparse
import statistics

from .confidence import ConfidenceGate, ConfidenceSettings, default_confidence_profiles, load_confidence_profiles, save_confidence_profiles
from .config import load_config
from .context import extract_lyric_context
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


def target_names(target: str) -> list[str]:
    return ["retrieval", "ngram", "transformer"] if target == "all" else [target]


def collect_retrieval_scores(generator: LyricGenerator, lines: list[str], max_samples: int) -> list[float]:
    scores: list[float] = []
    for line in lines[:max_samples]:
        result = generator.retriever.find_next_line(extract_lyric_context(context_from_line(line)))
        if result is not None and result.confidence > 0:
            scores.append(result.confidence)
    return scores


def collect_ngram_scores(generator: LyricGenerator, lines: list[str], max_samples: int) -> list[float]:
    model = generator.load_ngram_model()
    if model is None:
        return []
    scores: list[float] = []
    for line in lines[:max_samples]:
        result = model.predict(
            extract_lyric_context(context_from_line(line)),
            max_chars=generator.config.model.max_new_tokens,
            allow_fuzzy=True,
        )
        if result is not None and result.confidence > 0:
            scores.append(result.confidence)
    return scores


def collect_transformer_scores(generator: LyricGenerator, lines: list[str], max_samples: int) -> list[float]:
    try:
        generator.load()
    except Exception as exc:
        print({"target": "transformer", "warning": f"skipped transformer calibration: {exc}"})
        return []
    gate = ConfidenceGate(ConfidenceSettings(threshold=0.0, min_token_probability=0.0, max_repeat_ratio=1.0))
    scores: list[float] = []
    for line in lines[:max_samples]:
        prediction = generator._predict_loaded(extract_lyric_context(context_from_line(line)), gate)
        if prediction.confidence > 0:
            scores.append(prediction.confidence)
    return scores


def collect_scores(generator: LyricGenerator, target: str, lines: list[str], max_samples: int) -> list[float]:
    if target == "retrieval":
        return collect_retrieval_scores(generator, lines, max_samples)
    if target == "ngram":
        return collect_ngram_scores(generator, lines, max_samples)
    if target == "transformer":
        return collect_transformer_scores(generator, lines, max_samples)
    raise ValueError(f"Unsupported calibration target: {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate confidence threshold from validation lyric contexts.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--target", choices=("retrieval", "ngram", "transformer", "all"), default="all")
    parser.add_argument("--max-samples", type=int, default=64)
    args = parser.parse_args()
    config = load_config(args.config)
    valid_path = config.paths.processed_dir / "valid.txt"
    if not valid_path.exists():
        raise FileNotFoundError(f"Validation file not found: {valid_path}")
    lines = [line.strip() for line in valid_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError("Validation file is empty.")

    defaults = ConfidenceSettings(
        threshold=config.confidence.threshold,
        min_token_probability=config.confidence.min_token_probability,
        max_repeat_ratio=config.confidence.max_repeat_ratio,
    )
    existing = load_confidence_profiles(config.paths.model_dir, defaults)
    profiles = dict(existing.profiles)
    default_profiles = default_confidence_profiles(defaults)
    all_samples: dict[str, list[float]] = {}
    generator = LyricGenerator(config)

    for target in target_names(args.target):
        scores = collect_scores(generator, target, lines, args.max_samples)
        base = profiles.get(target, default_profiles[target])
        threshold = (
            percentile(scores, config.confidence.calibration_percentile)
            if len(scores) >= 8
            else base.threshold
        )
        profiles[target] = ConfidenceSettings(
            threshold=threshold,
            min_token_probability=base.min_token_probability,
            max_repeat_ratio=base.max_repeat_ratio,
        )
        all_samples[target] = scores
        mean = statistics.mean(scores) if scores else 0.0
        print({"target": target, "threshold": threshold, "samples": len(scores), "mean": mean})

    save_confidence_profiles(config.paths.model_dir, profiles, all_samples, legacy=existing.legacy)


if __name__ == "__main__":
    main()
