from pathlib import Path

from lyricpredict.cleaner import CleanedSong
from lyricpredict.confidence import ConfidenceSettings, save_confidence_profiles
from lyricpredict.config import AppConfig, ConfidenceConfig, InferenceConfig, ModelConfig, PathsConfig, TrainingConfig
from lyricpredict.generation import LyricGenerator, TransformerCandidate
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


def save_simple_ngram(config: AppConfig) -> None:
    ngram = CharNGramModel.train([CleanedSong(source="song", lines=["春天来了", "我们唱歌", "明天见面"])], order=8)
    ngram.save(config.paths.model_dir / "ngram_model.json")


def stub_transformer(monkeypatch, candidates: list[TransformerCandidate]) -> None:
    monkeypatch.setattr(
        LyricGenerator,
        "_generate_transformer_candidates",
        lambda self, context, policy: candidates,
    )


def test_model_only_uses_transformer_candidate_with_ngram_verifier(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    save_simple_ngram(config)
    stub_transformer(monkeypatch, [TransformerCandidate(text="，明天见面", confidence=0.95, reason="accepted")])

    def fail_if_called(*args, **kwargs):
        raise AssertionError("retrieval must not be called in model-only mode")

    monkeypatch.setattr("lyricpredict.retrieval.LyricRetriever.find_next_line", fail_if_called)
    prediction = LyricGenerator(config, mode="model-only").predict("春天来了，我们唱歌")

    assert prediction.accepted
    assert prediction.reason == "verified_transformer:ngram_exact"
    assert prediction.text == "，明天见面"


def test_model_only_does_not_call_legacy_ngram_predict(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    save_simple_ngram(config)
    stub_transformer(monkeypatch, [TransformerCandidate(text="，明天见面", confidence=0.95, reason="accepted")])

    def fail_if_called(*args, **kwargs):
        raise AssertionError("legacy n-gram generation must not be called by the main flow")

    monkeypatch.setattr(CharNGramModel, "predict", fail_if_called)
    prediction = LyricGenerator(config, mode="model-only").predict("春天来了，我们唱歌")

    assert prediction.accepted
    assert prediction.reason == "verified_transformer:ngram_exact"


def test_model_only_abstains_when_transformer_candidate_is_unsupported(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    save_simple_ngram(config)
    stub_transformer(monkeypatch, [TransformerCandidate(text="，错误答案", confidence=0.95, reason="accepted")])

    prediction = LyricGenerator(config, mode="model-only").predict("春天来了，我们唱歌")

    assert not prediction.accepted
    assert prediction.reason == "low_final_confidence"


def test_model_only_can_return_corrected_context_from_ngram_verifier(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    ngram = CharNGramModel.train(
        [CleanedSong(source="song", lines=["我想要我想要你知道，不论这世界多糟糕，未来的你会光芒万丈", "而我也曾是你万分之一的光"])],
        order=32,
    )
    ngram.save(config.paths.model_dir / "ngram_model.json")
    stub_transformer(monkeypatch, [TransformerCandidate(text="，而我也曾是你万分之一的光", confidence=0.98, reason="accepted")])

    prediction = LyricGenerator(config, mode="model-only").predict(
        "我想要我想要你知道，不论这世界多糟糕，未来的你会光茫万丈",
        strictness="tolerant",
        correction=True,
    )

    assert prediction.accepted
    assert prediction.corrected_context == "我想要我想要你知道，不论这世界多糟糕，未来的你会光芒万丈"


def test_strictness_parameter_affects_final_gate(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    save_simple_ngram(config)
    stub_transformer(monkeypatch, [TransformerCandidate(text="，明天见面", confidence=0.45, reason="accepted")])

    strict = LyricGenerator(config, mode="model-only").predict("春天来了，我们唱歌", strictness="strict")
    tolerant = LyricGenerator(config, mode="model-only").predict("春天来了，我们唱歌", strictness="tolerant")

    assert not strict.accepted
    assert tolerant.accepted


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
