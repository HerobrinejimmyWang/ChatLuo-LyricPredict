from pathlib import Path

from lyricpredict.cleaner import CleanedSong
from lyricpredict.config import AppConfig, ConfidenceConfig, InferenceConfig, ModelConfig, PathsConfig, TrainingConfig
from scripts.evaluate_testresult import (
    EvalResult,
    build_complex_context_cases,
    build_correction_pool,
    build_half_sentence_pool,
    build_multi_sentence_cases,
    build_single_sentence_cases,
    exact_match,
    format_expected,
    is_wrong_output,
    parse_strictnesses,
    render_report,
    result_column,
)


def make_config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        paths=PathsConfig(
            raw_dir=tmp_path / "data" / "raw",
            processed_dir=tmp_path / "data" / "processed",
            model_dir=tmp_path / "models" / "default",
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


def test_single_sentence_cases_remove_ambiguous_contexts():
    songs = [
        CleanedSong(source="a", lines=["同一句歌词", "第一种后续"]),
        CleanedSong(source="b", lines=["同一句歌词", "第二种后续"]),
        CleanedSong(source="c", lines=["唯一一句歌词", "唯一后续"]),
    ]

    cases = build_single_sentence_cases(songs)

    assert [case.input_text for case in cases] == ["唯一一句歌词"]
    assert cases[0].expected == "，唯一后续"


def test_multi_sentence_cases_extend_to_minimum_token_count():
    songs = [CleanedSong(source="song", lines=["一二三四", "五六七八", "九十十一十二", "下一句歌词"])]

    cases = build_multi_sentence_cases(songs, token_count=len, min_tokens=16)

    assert cases[-1].input_text == "一二三四，五六七八，九十十一十二"
    assert cases[-1].expected == "，下一句歌词"


def test_complex_context_cases_are_deterministic():
    songs = [CleanedSong(source="song", lines=["当前歌词", "下一句歌词"])]

    first = build_complex_context_cases(songs, seed=123)
    second = build_complex_context_cases(songs, seed=123)

    assert first == second
    assert first[0].input_text.endswith("当前歌词")
    assert first[0].input_text != "当前歌词"


def test_half_sentence_pool_cuts_inside_clause_without_extra_punctuation():
    songs = [CleanedSong(source="song", lines=["不论这世界多糟糕，未来的你会光芒万丈"])]

    cases = build_half_sentence_pool(songs)

    assert any(case.input_text == "不论这世" and case.expected == "界多糟糕" for case in cases)


def test_correction_pool_applies_requested_number_of_typos():
    songs = [CleanedSong(source="song", lines=["将故事传颂吧，未来的你会光芒万丈", "下一句歌词"])]

    one = build_correction_pool(songs, 1)
    two = build_correction_pool(songs, 2)

    assert one[0].input_text == "将故事传诵吧，未来的你会光芒万丈"
    assert two[0].input_text == "将故事传诵吧，未来的你会光茫万丈"
    assert one[0].expected == "，下一句歌词"


def test_exact_match_does_not_normalize_punctuation_or_spaces():
    assert exact_match("，下一句", "，下一句", True)
    assert not exact_match("，下一句", "下一句", True)
    assert not exact_match("，下一句", "，下一 句", True)
    assert not exact_match("，下一句", "，下一句", False)


def test_wrong_output_counts_only_accepted_mismatches():
    assert is_wrong_output("，下一句", "下一句", True)
    assert not is_wrong_output("，下一句", "，下一句", True)
    assert not is_wrong_output("，下一句", "", False)


def test_format_expected_respects_existing_boundary():
    assert format_expected("当前歌词", "下一句，后续") == "，下一句"
    assert format_expected("当前歌词，", "下一句，后续") == "下一句"


def test_result_report_includes_strictness_and_wrong_output_tables():
    columns = [result_column("auto", "strict"), result_column("auto", "balanced")]
    results = {
        "Single sentence": {
            "auto:strict": EvalResult(correct=1, wrong=0, total=2),
            "auto:balanced": EvalResult(correct=1, wrong=1, total=2),
        }
    }

    report = render_report(results, None, columns)

    assert "auto:strict" in report
    assert "### Accuracy" in report
    assert "### Wrong Outputs" in report
    assert "### Abstain" in report
    assert "1/2 (50.0%)" in report


def test_parse_strictnesses_accepts_three_ui_levels():
    assert parse_strictnesses("strict,balanced,tolerant") == ["strict", "balanced", "tolerant"]
