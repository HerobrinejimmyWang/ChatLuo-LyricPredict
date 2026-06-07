from pathlib import Path

from lyricpredict.cleaner import CleanedSong
from lyricpredict.config import AppConfig, ConfidenceConfig, InferenceConfig, ModelConfig, PathsConfig, TrainingConfig
from lyricpredict.generation import LyricGenerator
from lyricpredict.ngram_model import CharNGramModel


def make_config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        paths=PathsConfig(
            raw_dir=tmp_path / "raw",
            processed_dir=tmp_path / "processed",
            model_dir=tmp_path / "model",
            web_dir=tmp_path / "web",
        ),
        model=ModelConfig(
            base_model="unused",
            device="cpu",
            max_input_tokens=32,
            max_new_tokens=32,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.0,
            generation_attempts=1,
        ),
        training=TrainingConfig(
            block_size=32,
            validation_ratio=0.1,
            num_train_epochs=1,
            learning_rate=0.001,
            batch_size=1,
            gradient_accumulation_steps=1,
            lora_r=4,
            lora_alpha=8,
            lora_dropout=0.0,
        ),
        confidence=ConfidenceConfig(
            threshold=0.05,
            min_token_probability=0.0,
            max_repeat_ratio=0.5,
            calibration_percentile=20,
        ),
        inference=InferenceConfig(mode="model-only", model_fallback_after_retrieval=True),
    )


def test_model_only_uses_exported_model_artifact_without_retrieval(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    ngram = CharNGramModel.train([CleanedSong(source="song", lines=["将故事传颂吧", "风携它远追", "你脸颊热泪"])], order=8)
    ngram.save(config.paths.model_dir / "ngram_model.json")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("retrieval must not be called in model-only mode")

    monkeypatch.setattr("lyricpredict.retrieval.LyricRetriever.find_next_line", fail_if_called)
    prediction = LyricGenerator(config, mode="model-only").predict("将故事传颂吧，风携它远追")

    assert prediction.accepted
    assert prediction.reason == "char_ngram"
    assert prediction.text == "，你脸颊热泪"


def test_model_only_abstains_when_exported_artifact_misses(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    ngram = CharNGramModel.train([CleanedSong(source="song", lines=["将故事传颂吧", "风携它远追", "你脸颊热泪"])], order=8)
    ngram.save(config.paths.model_dir / "ngram_model.json")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("transformer fallback must not be called when exported artifact exists")

    monkeypatch.setattr(LyricGenerator, "load", fail_if_called)
    prediction = LyricGenerator(config, mode="model-only").predict("完全不相关的上下文")

    assert not prediction.accepted
    assert prediction.reason == "no_model_match"
