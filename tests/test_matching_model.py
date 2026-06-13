import json
from pathlib import Path

from lyricpredict.cleaner import CleanedSong
from lyricpredict.config import AppConfig, ConfidenceConfig, InferenceConfig, ModelConfig, PathsConfig, TrainingConfig
from lyricpredict.matching_model import (
    BigruRankerPredictor,
    CharMatchPredictor,
    LegacyNGramBenchmark,
    build_candidate_library,
    build_matching_samples,
    build_pair_examples,
    format_expected,
    load_matching_index,
    load_or_build_candidate_library,
    train_bigru_ranker,
    write_matching_index,
)
from lyricpredict.ngram_model import CharNGramModel
from scripts.evaluate_testresult import format_expected as result_format_expected


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
        inference=InferenceConfig(mode="auto", model_fallback_after_retrieval=True, strictness="balanced"),
    )


def write_songs(config: AppConfig, songs: list[CleanedSong]) -> None:
    config.paths.processed_dir.mkdir(parents=True)
    with (config.paths.processed_dir / "songs.jsonl").open("w", encoding="utf-8") as handle:
        for song in songs:
            handle.write(json.dumps({"source": song.source, "lines": song.lines}, ensure_ascii=False) + "\n")
    config.paths.model_dir.mkdir(parents=True)


def sample_songs() -> list[CleanedSong]:
    return [
        CleanedSong(source="song-a", lines=["春天来了", "我们唱歌", "明天见面"]),
        CleanedSong(source="song-b", lines=["月光落下", "星星回答", "风继续吹"]),
    ]


def test_matching_expected_matches_testresult_boundary_rule():
    assert format_expected("春天来了", "我们唱歌，后续") == result_format_expected("春天来了", "我们唱歌，后续")
    assert format_expected("春天来了，", "我们唱歌，后续") == result_format_expected("春天来了，", "我们唱歌，后续")


def test_candidate_library_skips_metadata_lines():
    songs = [
        CleanedSong(source="song", lines=["作词：某某", "春天来了", "我们唱歌", "版权声明：禁止转载", "明天见面"])
    ]

    candidates = build_candidate_library(songs)

    assert all("作词" not in candidate.text for candidate in candidates)
    assert all("版权" not in candidate.text for candidate in candidates)
    assert any(candidate.text for candidate in candidates)


def test_negative_sampling_is_reproducible():
    songs = sample_songs()
    samples = build_matching_samples(songs)
    candidates = build_candidate_library(songs)

    first = build_pair_examples(samples, candidates, seed=123)
    second = build_pair_examples(samples, candidates, seed=123)

    assert first == second
    assert any(example.label == 0 for example in first)
    assert any(example.label == 1 for example in first)


def test_char_match_predictor_does_not_call_legacy_ngram_predict(monkeypatch):
    songs = sample_songs()
    predictor = CharMatchPredictor(build_candidate_library(songs))

    def fail_if_called(*args, **kwargs):
        raise AssertionError("new matching predictor must not call legacy n-gram generation")

    monkeypatch.setattr(CharNGramModel, "predict", fail_if_called)
    prediction = predictor.predict("春天来了")

    assert prediction.accepted
    assert prediction.text == "，我们唱歌"


def test_char_match_strips_leading_separator_after_input_separator():
    songs = [CleanedSong(source="song", lines=["当前歌词", "下一句歌词"])]
    predictor = CharMatchPredictor(build_candidate_library(songs))

    prediction = predictor.predict("当前歌词，")

    assert prediction.accepted
    assert prediction.text == "下一句歌词"


def test_char_match_returns_corrected_context_when_requested():
    songs = [CleanedSong(source="song", lines=["将故事传颂吧", "下一句歌词"])]
    predictor = CharMatchPredictor(build_candidate_library(songs))

    without_notice = predictor.predict("将故事传诵吧", strictness="balanced", correction=False)
    with_notice = predictor.predict("将故事传诵吧", strictness="balanced", correction=True)

    assert without_notice.accepted
    assert without_notice.corrected_context is None
    assert with_notice.accepted
    assert with_notice.text == "，下一句歌词"
    assert with_notice.corrected_context == "将故事传颂吧"


def test_char_match_rejects_mixed_context_from_different_songs():
    songs = [
        CleanedSong(source="a", lines=["春天来了我们唱歌", "明天见面"]),
        CleanedSong(source="b", lines=["月光落下星星回答", "风继续吹"]),
    ]
    predictor = CharMatchPredictor(build_candidate_library(songs))

    prediction = predictor.predict("春天来了星星回答", strictness="balanced")

    assert not prediction.accepted


def test_char_match_rejects_out_of_library_text():
    songs = [CleanedSong(source="song", lines=["春天来了我们唱歌", "明天见面"])]
    predictor = CharMatchPredictor(build_candidate_library(songs))

    prediction = predictor.predict("这是一段普通说明文字不是歌词", strictness="balanced")

    assert not prediction.accepted


def test_matching_index_roundtrip_and_stale_detection(tmp_path):
    config = make_config(tmp_path)
    write_songs(config, sample_songs())

    written = write_matching_index(config.paths.processed_dir, sample_songs())
    loaded = load_matching_index(config.paths.processed_dir)

    assert loaded == written
    with (config.paths.processed_dir / "songs.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"source": "new", "lines": ["new line"]}, ensure_ascii=False) + "\n")
    assert load_matching_index(config.paths.processed_dir) is None


def test_load_or_build_candidate_library_creates_matching_index(tmp_path):
    config = make_config(tmp_path)
    write_songs(config, sample_songs())

    candidates = load_or_build_candidate_library(config.paths.processed_dir)

    assert candidates
    assert (config.paths.processed_dir / "matching_index.json").exists()


def test_char_match_prefers_half_sentence_candidate_with_spaces():
    songs = [CleanedSong(source="song", lines=["匠心不朽 承千载守望 Ha～", "穿行于璀璨 手可触星光"])]
    predictor = CharMatchPredictor(build_candidate_library(songs))

    prediction = predictor.predict("匠心不朽 承千", strictness="balanced")

    assert prediction.accepted
    assert prediction.text == "载守望 Ha～"


def test_char_match_prefers_exact_half_context_over_spacing_variant():
    songs = [
        CleanedSong(source="a", lines=["人太多大 部分是漫无目的的走"]),
        CleanedSong(source="b", lines=["人太多 大部分是漫无目的地走"]),
    ]
    predictor = CharMatchPredictor(build_candidate_library(songs))

    prediction = predictor.predict("人太多 大部分", strictness="balanced")

    assert prediction.accepted
    assert prediction.text == "是漫无目的地走"


def test_legacy_ngram_benchmark_calls_legacy_predict(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    write_songs(config, sample_songs())
    ngram = CharNGramModel.train(sample_songs(), order=8, min_context=2)
    ngram.save(config.paths.model_dir / "ngram_model.json")
    called = {"value": False}
    original = CharNGramModel.predict

    def wrapped(self, *args, **kwargs):
        called["value"] = True
        return original(self, *args, **kwargs)

    monkeypatch.setattr(CharNGramModel, "predict", wrapped)
    prediction = LegacyNGramBenchmark(config).predict("春天来了")

    assert called["value"]
    assert prediction.accepted


def test_char_bigru_trains_and_predicts_on_tiny_fixture(tmp_path):
    config = make_config(tmp_path)
    write_songs(config, sample_songs())

    stats = train_bigru_ranker(config, "char-bigru", epochs=1, batch_size=2, max_pairs=4)
    predictor = BigruRankerPredictor(config, "char-bigru", build_candidate_library(sample_songs()))
    prediction = predictor.predict("春天来了", strictness="tolerant")

    assert stats.examples > 0
    assert prediction.confidence >= 0.0


def test_token_bigru_trains_and_predicts_on_tiny_fixture(tmp_path, monkeypatch):
    class FakeTokenizer:
        def encode(self, text, add_special_tokens=False):
            return [ord(char) % 97 for char in text if not char.isspace()]

    config = make_config(tmp_path)
    write_songs(config, sample_songs())
    monkeypatch.setattr("lyricpredict.matching_model._load_hf_tokenizer", lambda tokenizer_name: FakeTokenizer())
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *args, **kwargs: FakeTokenizer())

    stats = train_bigru_ranker(
        config,
        "token-bigru",
        tokenizer_name="fake-tokenizer",
        epochs=1,
        batch_size=2,
        max_pairs=4,
    )
    predictor = BigruRankerPredictor(config, "token-bigru", build_candidate_library(sample_songs()))
    prediction = predictor.predict("春天来了", strictness="tolerant")

    assert stats.examples > 0
    assert prediction.confidence >= 0.0
