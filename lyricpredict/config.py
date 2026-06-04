from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PathsConfig:
    raw_dir: Path
    processed_dir: Path
    model_dir: Path
    web_dir: Path


@dataclass(frozen=True)
class ModelConfig:
    base_model: str
    device: str
    max_input_tokens: int
    max_new_tokens: int
    temperature: float
    top_p: float
    repetition_penalty: float
    generation_attempts: int


@dataclass(frozen=True)
class TrainingConfig:
    block_size: int
    validation_ratio: float
    num_train_epochs: int
    learning_rate: float
    batch_size: int
    gradient_accumulation_steps: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float


@dataclass(frozen=True)
class ConfidenceConfig:
    threshold: float
    min_token_probability: float
    max_repeat_ratio: float
    calibration_percentile: int


@dataclass(frozen=True)
class AppConfig:
    paths: PathsConfig
    model: ModelConfig
    training: TrainingConfig
    confidence: ConfidenceConfig


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def load_config(config_path: str | Path = "configs/default.yaml") -> AppConfig:
    path = Path(config_path).resolve()
    base = path.parent.parent
    raw = _load_yaml(path)

    paths = raw.get("paths", {})
    model = raw.get("model", {})
    training = raw.get("training", {})
    confidence = raw.get("confidence", {})

    return AppConfig(
        paths=PathsConfig(
            raw_dir=_resolve(base, str(paths.get("raw_dir", "data/raw"))),
            processed_dir=_resolve(base, str(paths.get("processed_dir", "data/processed"))),
            model_dir=_resolve(base, str(paths.get("model_dir", "models/default"))),
            web_dir=_resolve(base, str(paths.get("web_dir", "web"))),
        ),
        model=ModelConfig(
            base_model=str(model.get("base_model", "uer/gpt2-distil-chinese-cluecorpussmall")),
            device=str(model.get("device", "cpu")),
            max_input_tokens=int(model.get("max_input_tokens", 192)),
            max_new_tokens=int(model.get("max_new_tokens", 48)),
            temperature=float(model.get("temperature", 0.85)),
            top_p=float(model.get("top_p", 0.92)),
            repetition_penalty=float(model.get("repetition_penalty", 1.08)),
            generation_attempts=int(model.get("generation_attempts", 4)),
        ),
        training=TrainingConfig(
            block_size=int(training.get("block_size", 192)),
            validation_ratio=float(training.get("validation_ratio", 0.1)),
            num_train_epochs=int(training.get("num_train_epochs", 3)),
            learning_rate=float(training.get("learning_rate", 2e-4)),
            batch_size=int(training.get("batch_size", 2)),
            gradient_accumulation_steps=int(training.get("gradient_accumulation_steps", 8)),
            lora_r=int(training.get("lora_r", 8)),
            lora_alpha=int(training.get("lora_alpha", 16)),
            lora_dropout=float(training.get("lora_dropout", 0.05)),
        ),
        confidence=ConfidenceConfig(
            threshold=float(confidence.get("threshold", 0.18)),
            min_token_probability=float(confidence.get("min_token_probability", 0.01)),
            max_repeat_ratio=float(confidence.get("max_repeat_ratio", 0.35)),
            calibration_percentile=int(confidence.get("calibration_percentile", 20)),
        ),
    )
