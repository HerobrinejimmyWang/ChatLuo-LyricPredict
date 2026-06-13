from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum

from lyricpredict.generation import Prediction, TERMINATORS, context_is_inside_clause, normalize_prediction_boundary
from lyricpredict.retrieval import _key
from lyricpredict.separators import LEADING_SEPARATOR_CHARS, strip_leading_separators

from .settings import AppSettings


class SuggestionEvent(str, Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    IGNORE = "ignore"
    TIMEOUT = "timeout"


class SuggestionKind(str, Enum):
    PREDICTION = "prediction"
    CORRECTION = "correction"
    CONTINUE = "continue"


@dataclass(frozen=True)
class SuggestionPayload:
    context: str
    text: str
    confidence: float
    reason: str
    corrected_context: str | None = None
    kind: SuggestionKind = SuggestionKind.PREDICTION
    inserted_text: str = ""


@dataclass
class SuggestionState:
    payload: SuggestionPayload | None = None
    visible: bool = False
    last_event: SuggestionEvent | None = None
    rejected_key: str | None = None
    rejected_until: float = 0.0

    def show(self, payload: SuggestionPayload) -> None:
        self.payload = payload
        self.visible = True
        self.last_event = None

    def accept(self) -> SuggestionPayload | None:
        payload = self.payload
        self.visible = False
        self.last_event = SuggestionEvent.ACCEPT
        self.payload = None
        return payload

    def reject(self, cooldown_seconds: float = 8.0, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        self.rejected_key = _key(self.payload.context) if self.payload else None
        self.rejected_until = now + cooldown_seconds
        self.visible = False
        self.last_event = SuggestionEvent.REJECT
        self.payload = None

    def ignore(self) -> None:
        self.visible = False
        self.last_event = SuggestionEvent.IGNORE
        self.payload = None

    def timeout(self) -> None:
        self.visible = False
        self.last_event = SuggestionEvent.TIMEOUT
        self.payload = None

    def is_rejected(self, context: str, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        return bool(self.rejected_key and now < self.rejected_until and _key(context) == self.rejected_key)


def prediction_to_payload(context: str, prediction: Prediction) -> SuggestionPayload | None:
    if not prediction.accepted or not prediction.text:
        return None
    return SuggestionPayload(
        context=context,
        text=prediction.text,
        confidence=prediction.confidence,
        reason=prediction.reason,
        corrected_context=prediction.corrected_context,
    )


def display_reason(reason: str) -> str:
    if reason.startswith("char_match_half"):
        return "partial matched"
    if reason in {"char_match_suffix", "char_match_prefix", "char_match_overlap"}:
        return "matched"
    if reason == "char_match_ambiguous":
        return "ambiguous match"
    if reason == "char_match_threshold":
        return "low confidence"
    if reason == "char_match_no_candidate":
        return "no match"
    if reason == "retrieval":
        return "matched"
    if reason.startswith("verified_transformer:ngram_exact"):
        return "verified"
    if reason.startswith("verified_transformer:ngram_fuzzy"):
        return "verified with correction"
    if reason == "low_final_confidence":
        return "low confidence"
    if reason == "no_transformer_candidate":
        return "no usable output"
    if reason == "no_model_match":
        return "no match"
    return reason.replace("_", " ")


def _without_leading_separator(text: str) -> str:
    return strip_leading_separators(text)


def _configured_separator(settings: AppSettings) -> str:
    return settings.default_separator.replace("\\n", "\n")


def compose_insert_text(payload: SuggestionPayload, settings: AppSettings, replace_context: bool = False) -> str:
    boundary_context = payload.corrected_context or payload.context
    suggestion = normalize_prediction_boundary(boundary_context, payload.text)
    if not settings.include_separator:
        suggestion = _without_leading_separator(suggestion)
    elif boundary_context.rstrip()[-1:] in TERMINATORS or context_is_inside_clause(boundary_context):
        suggestion = _without_leading_separator(suggestion) if suggestion[:1] in LEADING_SEPARATOR_CHARS else suggestion
    else:
        suggestion = f"{_configured_separator(settings)}{_without_leading_separator(suggestion)}"

    if replace_context and payload.corrected_context:
        return f"{payload.corrected_context}{normalize_prediction_boundary(payload.corrected_context, suggestion)}"
    return suggestion


def trim_context(text: str, context_window: int) -> str:
    text = text.strip()
    if len(text) <= context_window:
        return text
    return text[-context_window:]


def describe_correction(context: str, corrected_context: str) -> str:
    if context == corrected_context:
        return ""
    left = 0
    limit = min(len(context), len(corrected_context))
    while left < limit and context[left] == corrected_context[left]:
        left += 1
    right_context = len(context)
    right_corrected = len(corrected_context)
    while right_context > left and right_corrected > left and context[right_context - 1] == corrected_context[right_corrected - 1]:
        right_context -= 1
        right_corrected -= 1
    before = context[left:right_context] or "∅"
    after = corrected_context[left:right_corrected] or "∅"
    return f"纠错：{before} -> {after}"
