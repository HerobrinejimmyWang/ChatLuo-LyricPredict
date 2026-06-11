from pathlib import Path

from lyricpredict.cleaner import CleanedSong
from lyricpredict.confidence import ConfidenceSettings, save_confidence_profiles
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
        inference=InferenceConfig(mode="model-only", model_fallback_after_retrieval=True, strictness="balanced"),
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


def test_model_only_uses_ngram_profile_not_retrieval_threshold(tmp_path):
    config = make_config(tmp_path)
    ngram = CharNGramModel.train([CleanedSong(source="song", lines=["春天来了", "我们唱歌", "明天见面"])], order=8)
    ngram.save(config.paths.model_dir / "ngram_model.json")
    save_confidence_profiles(
        config.paths.model_dir,
        {
            "retrieval": ConfidenceSettings(threshold=1.0, min_token_probability=0.0, max_repeat_ratio=0.35),
            "ngram": ConfidenceSettings(threshold=0.0, min_token_probability=0.0, max_repeat_ratio=0.9),
            "transformer": ConfidenceSettings(threshold=1.0, min_token_probability=0.0, max_repeat_ratio=0.35),
        },
    )

    prediction = LyricGenerator(config, mode="model-only").predict("春天来了，我们唱歌")

    assert prediction.accepted
    assert prediction.reason == "char_ngram"


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


def test_model_only_tolerant_does_not_fallback_when_exported_artifact_misses(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    ngram = CharNGramModel.train([CleanedSong(source="song", lines=["将故事传颂吧", "风携它远追", "你脸颊热泪"])], order=8)
    ngram.save(config.paths.model_dir / "ngram_model.json")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("transformer fallback must not be called when exported artifact exists")

    monkeypatch.setattr(LyricGenerator, "load", fail_if_called)
    prediction = LyricGenerator(config, mode="model-only").predict("完全不相关的上下文", strictness="tolerant")

    assert not prediction.accepted
    assert prediction.reason == "no_model_match"


def test_model_only_can_return_corrected_context_when_enabled(tmp_path):
    config = make_config(tmp_path)
    ngram = CharNGramModel.train(
        [CleanedSong(source="song", lines=["不论这世界多糟糕，未来的你会光芒万丈", "而我也曾是你万分之一的光"])],
        order=16,
    )
    ngram.save(config.paths.model_dir / "ngram_model.json")

    generator = LyricGenerator(config, mode="model-only")
    without_correction = generator.predict("不论这世界多糟糕，未来的你会光茫万丈", correction=False)
    with_correction = generator.predict("不论这世界多糟糕，未来的你会光茫万丈", correction=True)

    assert without_correction.accepted
    assert without_correction.corrected_context is None
    assert with_correction.accepted
    assert with_correction.corrected_context == "不论这世界多糟糕，未来的你会光芒万丈"


def test_strictness_parameter_is_accepted_by_generator(tmp_path):
    config = make_config(tmp_path)
    ngram = CharNGramModel.train([CleanedSong(source="song", lines=["三人行，必有我师焉。"])], order=8, min_context=2)
    ngram.save(config.paths.model_dir / "ngram_model.json")

    prediction = LyricGenerator(config, mode="model-only").predict("三人行，", strictness="tolerant")

    assert prediction.accepted


def test_auto_retrieval_can_return_corrected_context_when_enabled(tmp_path):
    config = make_config(tmp_path)
    extra_dir = config.paths.processed_dir.parent.parent / "selflyricdata"
    extra_dir.mkdir()
    save_confidence_profiles(
        config.paths.model_dir,
        {
            "retrieval": ConfidenceSettings(threshold=0.0, min_token_probability=0.0, max_repeat_ratio=0.9),
            "ngram": ConfidenceSettings(threshold=0.0, min_token_probability=0.0, max_repeat_ratio=0.9),
            "transformer": ConfidenceSettings(threshold=0.0, min_token_probability=0.0, max_repeat_ratio=0.9),
        },
    )
    (extra_dir / "song.lrc").write_text(
        "[00:01.00]未来的你会光芒万丈\n[00:02.00]而我也曾是你万分之一的光",
        encoding="utf-8",
    )

    prediction = LyricGenerator(config, mode="auto").predict(
        "未来的你会光茫万丈",
        correction=True,
    )

    assert prediction.accepted
    assert prediction.reason == "retrieval"
    assert prediction.corrected_context == "未来的你会光芒万丈"
