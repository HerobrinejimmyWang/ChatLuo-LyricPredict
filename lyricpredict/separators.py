from __future__ import annotations

HARD_TERMINATORS = (",", ".", "，", "。")
WHITESPACE_SEPARATORS = (" ", "\t", "\r", "\n", "\u3000")
SEMANTIC_SEPARATORS = HARD_TERMINATORS + WHITESPACE_SEPARATORS
LEADING_SEPARATOR_CHARS = "".join(SEMANTIC_SEPARATORS)


def ends_with_separator(text: str) -> bool:
    return bool(text) and text[-1:] in SEMANTIC_SEPARATORS


def starts_with_separator(text: str) -> bool:
    return bool(text) and text[:1] in SEMANTIC_SEPARATORS


def strip_leading_separators(text: str) -> str:
    return text.lstrip(LEADING_SEPARATOR_CHARS).strip()


def is_semantic_separator(char: str) -> bool:
    return char in SEMANTIC_SEPARATORS
