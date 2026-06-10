from lyricpredict.desktop.uia_context import (
    TEXT_PATTERN_RANGE_ENDPOINT_END,
    TEXT_PATTERN_RANGE_ENDPOINT_START,
    TEXT_UNIT_CHARACTER,
    text_before_range,
)


class MockTextRange:
    def __init__(self, text: str, start: int, end: int | None = None):
        self.text = text
        self.start = start
        self.end = start if end is None else end

    def Clone(self):
        return MockTextRange(self.text, self.start, self.end)

    def MoveEndpointByRange(self, endpoint, other, other_endpoint):
        assert endpoint == TEXT_PATTERN_RANGE_ENDPOINT_END
        assert other_endpoint == TEXT_PATTERN_RANGE_ENDPOINT_START
        self.end = other.start

    def MoveEndpointByUnit(self, endpoint, unit, count):
        assert endpoint == TEXT_PATTERN_RANGE_ENDPOINT_START
        assert unit == TEXT_UNIT_CHARACTER
        self.start = max(0, self.start + count)

    def GetText(self, max_count):
        return self.text[self.start : self.end][:max_count]


def test_text_before_range_reads_context_before_caret():
    text_range = MockTextRange("一二三四五六七八九十", start=10)

    assert text_before_range(text_range, 4) == "七八九十"


def test_text_before_range_collapses_selection_to_start():
    text_range = MockTextRange("一二三四五六七八九十", start=6, end=8)

    assert text_before_range(text_range, 4) == "三四五六"
