from lyricpredict.desktop.settings import AppSettings
from lyricpredict.desktop.workflow import (
    SuggestionEvent,
    SuggestionPayload,
    SuggestionState,
    compose_insert_text,
    describe_correction,
    trim_context,
)


def test_suggestion_state_accept_reject_ignore():
    state = SuggestionState()
    payload = SuggestionPayload(context="未来会光茫万丈", text="，而我也曾是光", confidence=0.9, reason="test")

    state.show(payload)
    assert state.visible
    assert state.accept() == payload
    assert state.last_event == SuggestionEvent.ACCEPT

    state.show(payload)
    state.reject(cooldown_seconds=10, now=1.0)
    assert state.last_event == SuggestionEvent.REJECT
    assert state.is_rejected("未来会光茫万丈", now=5.0)
    assert not state.is_rejected("其他上下文", now=5.0)

    state.show(payload)
    state.ignore()
    assert state.last_event == SuggestionEvent.IGNORE


def test_compose_insert_text_uses_correction_before_prediction():
    payload = SuggestionPayload(
        context="未来的你会光茫万丈",
        corrected_context="未来的你会光芒万丈",
        text="，而我也曾是你万分之一的光",
        confidence=0.9,
        reason="char_ngram_fuzzy",
    )

    assert compose_insert_text(payload, AppSettings(), replace_context=True) == "未来的你会光芒万丈，而我也曾是你万分之一的光"
    assert compose_insert_text(payload, AppSettings(), replace_context=False) == "，而我也曾是你万分之一的光"


def test_compose_insert_text_can_strip_separator():
    payload = SuggestionPayload(context="未来的你会光芒万丈", text="，而我也曾是光", confidence=0.9, reason="test")

    assert compose_insert_text(payload, AppSettings(include_separator=False)) == "而我也曾是光"


def test_compose_insert_text_uses_configured_default_separator():
    payload = SuggestionPayload(context="未来的你会光芒万丈", text="，而我也曾是光", confidence=0.9, reason="test")

    assert compose_insert_text(payload, AppSettings(default_separator="，")) == "，而我也曾是光"
    assert compose_insert_text(payload, AppSettings(default_separator="。")) == "。而我也曾是光"
    assert compose_insert_text(payload, AppSettings(default_separator=" ")) == " 而我也曾是光"
    assert compose_insert_text(payload, AppSettings(default_separator="\n")) == "\n而我也曾是光"
    assert compose_insert_text(payload, AppSettings(default_separator="\\n")) == "\n而我也曾是光"


def test_compose_insert_text_does_not_add_separator_inside_clause():
    payload = SuggestionPayload(context="我想要我想要你知道，不论", text="，这世界多糟糕", confidence=0.9, reason="test")

    assert compose_insert_text(payload, AppSettings(default_separator="，")) == "这世界多糟糕"
    assert compose_insert_text(payload, AppSettings(default_separator=" ")) == "这世界多糟糕"


def test_trim_context_keeps_last_window_characters():
    assert trim_context("一二三四五六", 4) == "三四五六"


def test_describe_correction_highlights_changed_span():
    assert describe_correction("未来的你会光茫万丈", "未来的你会光芒万丈") == "纠错：茫 -> 芒"
