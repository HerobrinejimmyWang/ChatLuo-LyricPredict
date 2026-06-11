from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

REPEATED_CJK_RE = re.compile(r"([\u4e00-\u9fff])\1{3,}")
HYPHEN_ARTIFACT_RE = re.compile(r"(?:-\s*){3,}")


@dataclass(frozen=True)
class ConfidenceSettings:
    threshold: float = 0.18
    min_token_probability: float = 0.01
    max_repeat_ratio: float = 0.35


CONFIDENCE_SOURCES = ("retrieval", "ngram", "transformer")


@dataclass(frozen=True)
class ConfidenceProfiles:
    profiles: dict[str, ConfidenceSettings]
    legacy: ConfidenceSettings | None = None
    samples: dict[str, list[float]] | None = None

    def profile(self, source: str) -> ConfidenceSettings:
        return self.profiles.get(source, self.profiles["transformer"])


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


def cjk_count(text: str) -> int:
    return sum(1 for char in text if "\u4e00" <= char <= "\u9fff")


def dominant_cjk_ratio(text: str) -> float:
    chars = [char for char in text if "\u4e00" <= char <= "\u9fff"]
    if not chars:
        return 0.0
    return max(chars.count(char) for char in set(chars)) / len(chars)


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
        if "##" in text:
            return ConfidenceResult(False, 0.0, "subword_artifact")
        if HYPHEN_ARTIFACT_RE.search(text):
            return ConfidenceResult(False, 0.0, "hyphen_artifact")
        if cjk_count(text) < 2:
            return ConfidenceResult(False, 0.0, "too_little_chinese")
        if REPEATED_CJK_RE.search(text):
            return ConfidenceResult(False, 0.0, "repeated_character")
        if cjk_count(text) >= 8 and dominant_cjk_ratio(text) > 0.35:
            return ConfidenceResult(False, 0.0, "dominant_character")
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


def default_confidence_profiles(defaults: ConfidenceSettings) -> dict[str, ConfidenceSettings]:
    return {
        "retrieval": ConfidenceSettings(threshold=0.90, min_token_probability=0.0, max_repeat_ratio=0.35),
        "ngram": ConfidenceSettings(threshold=0.05, min_token_probability=0.0, max_repeat_ratio=0.45),
        "transformer": ConfidenceSettings(
            threshold=defaults.threshold,
            min_token_probability=defaults.min_token_probability,
            max_repeat_ratio=defaults.max_repeat_ratio,
        ),
    }


def _settings_from_dict(data: dict, defaults: ConfidenceSettings) -> ConfidenceSettings:
    return ConfidenceSettings(
        threshold=float(data.get("threshold", defaults.threshold)),
        min_token_probability=float(data.get("min_token_probability", defaults.min_token_probability)),
        max_repeat_ratio=float(data.get("max_repeat_ratio", defaults.max_repeat_ratio)),
    )


def load_confidence_profiles(model_dir: Path, defaults: ConfidenceSettings) -> ConfidenceProfiles:
    path = model_dir / "confidence.json"
    profiles = default_confidence_profiles(defaults)
    if not path.exists():
        return ConfidenceProfiles(profiles=profiles)

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data.get("profiles"), dict):
        loaded: dict[str, ConfidenceSettings] = {}
        for source in CONFIDENCE_SOURCES:
            profile_data = data["profiles"].get(source, {})
            loaded[source] = _settings_from_dict(profile_data, profiles[source])
        legacy = _settings_from_dict(data["legacy"], defaults) if isinstance(data.get("legacy"), dict) else None
        samples = data.get("samples") if isinstance(data.get("samples"), dict) else None
        return ConfidenceProfiles(profiles=loaded, legacy=legacy, samples=samples)

    legacy = _settings_from_dict(data, defaults)
    profiles["retrieval"] = ConfidenceSettings(
        threshold=legacy.threshold,
        min_token_probability=legacy.min_token_probability,
        max_repeat_ratio=legacy.max_repeat_ratio,
    )
    return ConfidenceProfiles(profiles=profiles, legacy=legacy, samples={"legacy": data.get("samples", [])})


def load_confidence_settings(model_dir: Path, defaults: ConfidenceSettings) -> ConfidenceSettings:
    path = model_dir / "confidence.json"
    if not path.exists():
        return defaults
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data.get("profiles"), dict):
        legacy = data.get("legacy")
        if isinstance(legacy, dict):
            return _settings_from_dict(legacy, defaults)
        transformer = data["profiles"].get("transformer", {})
        return _settings_from_dict(transformer, defaults)
    return _settings_from_dict(data, defaults)


def save_confidence_profiles(
    model_dir: Path,
    profiles: dict[str, ConfidenceSettings],
    samples: dict[str, list[float]] | None = None,
    legacy: ConfidenceSettings | None = None,
) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    completed = default_confidence_profiles(profiles.get("transformer", ConfidenceSettings()))
    completed.update({source: settings for source, settings in profiles.items() if source in CONFIDENCE_SOURCES})
    payload = {
        "version": 2,
        "profiles": {source: asdict(completed[source]) for source in CONFIDENCE_SOURCES},
        "samples": {source: (samples or {}).get(source, []) for source in CONFIDENCE_SOURCES},
    }
    if legacy is not None:
        payload["legacy"] = asdict(legacy)
    (model_dir / "confidence.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_confidence_settings(model_dir: Path, settings: ConfidenceSettings, samples: list[float] | None = None) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    payload = asdict(settings)
    if samples is not None:
        payload["samples"] = samples
    (model_dir / "confidence.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
