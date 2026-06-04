from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ConfidenceSettings:
    threshold: float = 0.18
    min_token_probability: float = 0.01
    max_repeat_ratio: float = 0.35


@dataclass(frozen=True)
class ConfidenceResult:
    accepted: bool
    confidence: float
    reason: str


def repetition_ratio(text: str, n: int = 3) -> float:
    chars = [char for char in text if not char.isspace()]
    if len(chars) < n * 2:
        return 0.0
    grams = ["".join(chars[i : i + n]) for i in range(len(chars) - n + 1)]
    if not grams:
        return 0.0
    repeated = len(grams) - len(set(grams))
    return repeated / len(grams)


def score_candidate(token_probabilities: list[float], text: str) -> float:
    if not token_probabilities:
        return 0.0
    avg_logprob = sum(math.log(max(prob, 1e-12)) for prob in token_probabilities) / len(token_probabilities)
    avg_probability = math.exp(avg_logprob)
    min_probability = min(token_probabilities)
    repeat_penalty = repetition_ratio(text)
    return max(0.0, min(1.0, (0.75 * avg_probability + 0.25 * min_probability) * (1.0 - repeat_penalty)))


class ConfidenceGate:
    def __init__(self, settings: ConfidenceSettings):
        self.settings = settings

    def evaluate(self, text: str, token_probabilities: list[float], ended: bool) -> ConfidenceResult:
        if not ended:
            return ConfidenceResult(False, 0.0, "no_terminator")
        if not text.strip():
            return ConfidenceResult(False, 0.0, "empty")
        confidence = score_candidate(token_probabilities, text)
        min_probability = min(token_probabilities) if token_probabilities else 0.0
        repeat = repetition_ratio(text)
        if min_probability < self.settings.min_token_probability:
            return ConfidenceResult(False, confidence, "min_token_probability")
        if repeat > self.settings.max_repeat_ratio:
            return ConfidenceResult(False, confidence, "repetition")
        if confidence < self.settings.threshold:
            return ConfidenceResult(False, confidence, "threshold")
        return ConfidenceResult(True, confidence, "accepted")


def load_confidence_settings(model_dir: Path, defaults: ConfidenceSettings) -> ConfidenceSettings:
    path = model_dir / "confidence.json"
    if not path.exists():
        return defaults
    data = json.loads(path.read_text(encoding="utf-8"))
    return ConfidenceSettings(
        threshold=float(data.get("threshold", defaults.threshold)),
        min_token_probability=float(data.get("min_token_probability", defaults.min_token_probability)),
        max_repeat_ratio=float(data.get("max_repeat_ratio", defaults.max_repeat_ratio)),
    )


def save_confidence_settings(model_dir: Path, settings: ConfidenceSettings, samples: list[float] | None = None) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    payload = asdict(settings)
    if samples is not None:
        payload["samples"] = samples
    (model_dir / "confidence.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
