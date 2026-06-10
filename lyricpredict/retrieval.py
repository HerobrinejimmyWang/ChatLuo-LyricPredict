from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .cleaner import CleanedSong, clean_lyrics_file, is_credit_line

TERMINATORS = (",", ".", "，", "。")
LYRIC_SUFFIXES = {".txt", ".lrc"}
SPACE_RE = re.compile(r"\s+")
PUNCT_RE = re.compile(r"[,，.。;；:：!！?？、\"'“”‘’《》()\[\]（）\-—_]")
META_PREFIXES = ("作词", "作曲", "编曲", "制作人", "版权", "未经著作权人")
MIN_MATCH_COVERAGE = 0.7
MIN_CONTEXT_KEY_LENGTH = 5


@dataclass(frozen=True)
class RetrievalResult:
    text: str
    confidence: float
    reason: str
    corrected_context: str | None = None


def _key(text: str) -> str:
    text = SPACE_RE.sub("", text.lower())
    return PUNCT_RE.sub("", text)


def _key_positions(text: str) -> list[int]:
    positions: list[int] = []
    for index, char in enumerate(text):
        if _key(char):
            positions.append(index)
    return positions


def _correct_context_suffix(context: str, corrected_suffix: str, matched_key_length: int) -> str | None:
    positions = _key_positions(context)
    corrected_key = _key(corrected_suffix)
    if len(positions) < matched_key_length or len(corrected_key) != matched_key_length:
        return None
    start = positions[-matched_key_length]
    end = positions[-1] + 1
    suffix_chars = list(context[start:end])
    key_index = 0
    for index, char in enumerate(suffix_chars):
        if _key(char):
            suffix_chars[index] = corrected_key[key_index]
            key_index += 1
    corrected = f"{context[:start]}{''.join(suffix_chars)}{context[end:]}"
    return corrected if corrected != context else None


def _cut_at_terminator(text: str) -> str:
    positions = [text.find(mark) for mark in TERMINATORS if text.find(mark) >= 0]
    if not positions:
        return text.strip()
    return text[: min(positions)].strip()


def _is_usable_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if is_credit_line(stripped):
        return False
    if any(stripped.startswith(prefix) for prefix in META_PREFIXES):
        return False
    return sum(1 for char in stripped if "\u4e00" <= char <= "\u9fff") >= 2


def _edit_distance_limited(left: str, right: str, limit: int) -> int:
    if abs(len(left) - len(right)) > limit:
        return limit + 1
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        row_min = current[0]
        for right_index, right_char in enumerate(right, start=1):
            cost = 0 if left_char == right_char else 1
            value = min(previous[right_index] + 1, current[right_index - 1] + 1, previous[right_index - 1] + cost)
            current.append(value)
            row_min = min(row_min, value)
        if row_min > limit:
            return limit + 1
        previous = current
    return previous[-1]


def _allowed_errors(length: int) -> int:
    if length < MIN_CONTEXT_KEY_LENGTH:
        return 0
    if length < 8:
        return 1
    if length < 16:
        return 3
    if length < 34:
        return 4
    return 5


def _continuation_parts(line: str) -> list[str]:
    parts = [part.strip() for part in SPACE_RE.split(line) if _is_usable_line(part)]
    if len(parts) >= 2:
        return parts
    split_parts = [part.strip() for part in re.split(r"[,，.。]", line) if _is_usable_line(part)]
    return split_parts if len(split_parts) >= 2 else []


def _suffix_match_score(context_key: str, window_key: str) -> tuple[int, int] | None:
    if len(window_key) < MIN_CONTEXT_KEY_LENGTH:
        return None
    coverage = len(window_key) / max(1, len(context_key))
    if coverage < MIN_MATCH_COVERAGE:
        return None
    suffix = context_key[-len(window_key) :]
    if suffix == window_key:
        return len(window_key), 0

    allowed = _allowed_errors(len(window_key))
    distance = _edit_distance_limited(suffix, window_key, allowed)
    if distance <= allowed:
        return len(window_key), distance
    return None


def _iter_song_files(directory: Path) -> Iterable[Path]:
    if not directory.exists():
        return []
    return sorted(path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in LYRIC_SUFFIXES)


def _load_processed_songs(processed_dir: Path) -> list[CleanedSong]:
    songs_path = processed_dir / "songs.jsonl"
    if not songs_path.exists():
        return []
    songs: list[CleanedSong] = []
    with songs_path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            data = json.loads(raw)
            songs.append(CleanedSong(source=str(data.get("source", "processed")), lines=list(data.get("lines", []))))
    return songs


class LyricRetriever:
    def __init__(self, processed_dir: Path, extra_dirs: Iterable[Path] = (), max_window_lines: int = 8):
        self.processed_dir = processed_dir
        self.extra_dirs = tuple(extra_dirs)
        self.max_window_lines = max_window_lines
        self._songs: list[CleanedSong] | None = None

    def _load_songs(self) -> list[CleanedSong]:
        if self._songs is not None:
            return self._songs

        songs = _load_processed_songs(self.processed_dir)
        seen_sources = {song.source for song in songs}
        for directory in self.extra_dirs:
            for path in _iter_song_files(directory):
                if str(path) in seen_sources:
                    continue
                song = clean_lyrics_file(path)
                if song.lines:
                    songs.append(song)
                    seen_sources.add(str(path))

        self._songs = songs
        return songs

    def _format_next_text(
        self,
        context: str,
        next_text: str,
        corrected_context: str | None = None,
    ) -> RetrievalResult:
        next_text = _cut_at_terminator(next_text)
        prefix = "" if context.rstrip().endswith(TERMINATORS) or next_text.startswith(TERMINATORS) else "，"
        return RetrievalResult(
            text=f"{prefix}{next_text}",
            confidence=0.92,
            reason="retrieval",
            corrected_context=corrected_context,
        )

    def _find_intra_line_continuation(self, context: str, context_key: str) -> RetrievalResult | None:
        direct_matches: dict[str, int] = {}
        matches: dict[str, int] = {}
        corrections: dict[str, str | None] = {}
        best_score: tuple[int, int] = (0, 999)
        for song in self._load_songs():
            lines = [line.strip() for line in song.lines if _is_usable_line(line)]
            for line_index, line in enumerate(lines):
                line_key = _key(line)
                if line_key.startswith(context_key) and len(context_key) >= MIN_CONTEXT_KEY_LENGTH:
                    raw_context = context.strip().rstrip("".join(TERMINATORS))
                    if line.startswith(raw_context):
                        next_text = line[len(raw_context) :].strip()
                        if _is_usable_line(next_text):
                            direct_matches[next_text] = direct_matches.get(next_text, 0) + 1
                parts = _continuation_parts(line)
                if len(parts) < 2:
                    continue
                for split_at in range(1, len(parts)):
                    prefix_part = "".join(parts[:split_at])
                    next_text = " ".join(parts[split_at:])
                    if not _is_usable_line(next_text):
                        continue
                    max_window = min(self.max_window_lines - 1, line_index)
                    for window in range(0, max_window + 1):
                        start = line_index - window
                        window_key = _key("".join(lines[start:line_index]) + prefix_part)
                        score = _suffix_match_score(context_key, window_key)
                        if score is None:
                            continue
                        if score[0] > best_score[0] or (score[0] == best_score[0] and score[1] < best_score[1]):
                            matches = {}
                            corrections = {}
                            best_score = score
                        if score == best_score:
                            matches[next_text] = matches.get(next_text, 0) + 1
                            corrected_source = "".join(lines[start:line_index]) + prefix_part
                            corrections[next_text] = (
                                _correct_context_suffix(context, corrected_source, score[0]) if score[1] > 0 else None
                            )

        if direct_matches:
            if len(direct_matches) > 1:
                return None
            return RetrievalResult(text=next(iter(direct_matches)), confidence=0.92, reason="retrieval")
        if not matches or len(matches) > 1:
            return None
        next_text = next(iter(matches))
        return self._format_next_text(context, next_text, corrections.get(next_text))

    def find_next_line(self, context: str) -> RetrievalResult | None:
        context_key = _key(context)
        if len(context_key) < MIN_CONTEXT_KEY_LENGTH:
            return None

        matches: dict[str, int] = {}
        corrections: dict[str, str | None] = {}
        best_score: tuple[int, int] = (0, 999)
        for song in self._load_songs():
            lines = [line.strip() for line in song.lines if _is_usable_line(line)]
            for next_index in range(1, len(lines)):
                max_window = min(self.max_window_lines, next_index)
                for window in range(1, max_window + 1):
                    start = next_index - window
                    window_key = _key("".join(lines[start:next_index]))
                    if len(window_key) < best_score[0]:
                        continue
                    score = _suffix_match_score(context_key, window_key)
                    if score is None:
                        continue
                    next_line = lines[next_index]
                    if not _is_usable_line(next_line):
                        continue
                    if score[0] > best_score[0] or (score[0] == best_score[0] and score[1] < best_score[1]):
                        matches = {}
                        corrections = {}
                        ordered_matches = []
                        best_score = score
                    if score == best_score:
                        matches[next_line] = matches.get(next_line, 0) + 1
                        corrected_source = "".join(lines[start:next_index])
                        corrections[next_line] = (
                            _correct_context_suffix(context, corrected_source, score[0]) if score[1] > 0 else None
                        )
                        ordered_matches.append(next_line)

        if not matches:
            return self._find_intra_line_continuation(context, context_key)
        if len(matches) > 1:
            if best_score[0] < 24 or len(context_key) - best_score[0] > 4:
                return None
            next_line = ordered_matches[-1]
            result = self._format_next_text(context, next_line, corrections.get(next_line))
            confidence = min(0.9, 0.5 + best_score[0] / 90 - best_score[1] * 0.04)
            return RetrievalResult(
                text=result.text,
                confidence=confidence,
                reason=result.reason,
                corrected_context=result.corrected_context,
            )

        next_line = next(iter(matches))
        confidence = min(0.99, 0.5 + best_score[0] / 80 - best_score[1] * 0.04)
        result = self._format_next_text(context, next_line, corrections.get(next_line))
        return RetrievalResult(
            text=result.text,
            confidence=confidence,
            reason=result.reason,
            corrected_context=result.corrected_context,
        )
