from __future__ import annotations

import re

CJK_RE = re.compile(r"[\u4e00-\u9fff]")
OPEN_QUOTES = ("“", "\"", "‘", "'")
CLOSE_QUOTES = ("”", "\"", "’", "'")
BOUNDARY_MARKS = ("：", ":")


def cjk_count(text: str) -> int:
    return len(CJK_RE.findall(text))


def _suffix_after_last(text: str, marks: tuple[str, ...], forbidden: tuple[str, ...] = ()) -> str | None:
    positions = [text.rfind(mark) for mark in marks]
    position = max(positions)
    if position < 0:
        return None
    suffix = text[position + 1 :].strip()
    if any(mark in suffix for mark in forbidden):
        return None
    return suffix if cjk_count(suffix) >= 4 else None


def extract_lyric_context(text: str) -> str:
    """Extract the most likely lyric suffix from mixed prose.

    Real inputs often quote lyrics inside commentary. The predictor only needs
    the lyric suffix, so use the last quoted or colon-introduced segment when it
    contains enough Chinese text.
    """
    text = text.strip()
    quoted = _suffix_after_last(text, OPEN_QUOTES, CLOSE_QUOTES)
    if quoted is not None:
        return quoted
    colon_suffix = _suffix_after_last(text, BOUNDARY_MARKS)
    if colon_suffix is not None:
        return colon_suffix
    return text
