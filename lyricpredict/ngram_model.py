from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from .cleaner import CleanedSong
from .config import load_config
from .retrieval import META_PREFIXES, TERMINATORS, _is_usable_line, _key, _load_processed_songs
from .separators import ends_with_separator, is_semantic_separator, starts_with_separator, strip_leading_separators

EDGE_QUOTES = "\"'“”‘’"


@dataclass(frozen=True)
class NGramPrediction:
    text: str
    confidence: float
    reason: str
    corrected_context: str | None = None


@dataclass(frozen=True)
class NGramVerification:
    score: float
    reason: str
    corrected_context: str | None = None


def _append_with_boundary(pieces: list[str], line: str) -> None:
    if not pieces:
        pieces.append(line)
        return
    separator = "" if ends_with_separator(pieces[-1]) or starts_with_separator(line) else "，"
    pieces.append(f"{separator}{line}")


def song_to_training_text(song: CleanedSong) -> str:
    pieces: list[str] = []
    for raw_line in song.lines:
        line = raw_line.strip()
        if not _is_usable_line(line):
            continue
        if any(line.startswith(prefix) for prefix in META_PREFIXES):
            continue
        _append_with_boundary(pieces, line)
    text = "".join(pieces)
    return text if text.endswith(TERMINATORS) else f"{text}。"


def _has_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def next_continuation(text: str, start: int, max_chars: int = 80) -> str:
    generated: list[str] = []
    for char in text[start : start + max_chars]:
        generated.append(char)
        if is_semantic_separator(char) and _has_cjk("".join(generated)):
            return "".join(generated[:-1]).strip().strip(EDGE_QUOTES)
    return ""


class CharNGramModel:
    def __init__(
        self,
        order: int,
        transitions: dict[str, dict[str, int]],
        min_context: int = 8,
        continuations: dict[str, dict[str, int]] | None = None,
    ):
        self.order = order
        self.transitions = transitions
        self.min_context = min_context
        self.continuations = continuations or {}
        self._normalized_continuation_buckets: dict[int, dict[str, Counter[str]]] | None = None
        self._normalized_context_forms: dict[str, Counter[str]] | None = None

    @classmethod
    def train(
        cls,
        songs: list[CleanedSong],
        order: int = 16,
        min_count: int = 1,
        min_context: int = 8,
    ) -> "CharNGramModel":
        counters: dict[str, Counter[str]] = defaultdict(Counter)
        continuations: dict[str, Counter[str]] = defaultdict(Counter)
        for song in songs:
            text = song_to_training_text(song)
            for index, char in enumerate(text):
                max_length = min(order * 3, index)
                for length in range(1, max_length + 1):
                    raw_context = text[index - length : index]
                    context = _key(raw_context)
                    if not context or len(context) > order:
                        continue
                    counters[context][char] += 1
                    continuation = next_continuation(text, index)
                    if continuation:
                        continuations[raw_context][continuation] += 1
        transitions = {
            context: dict(counter)
            for context, counter in counters.items()
            if sum(counter.values()) >= min_count and len(context) <= order
        }
        continuation_payload = {
            context: dict(counter)
            for context, counter in continuations.items()
            if _key(context) in transitions and sum(counter.values()) >= min_count
        }
        return cls(order=order, transitions=transitions, min_context=min_context, continuations=continuation_payload)

    @classmethod
    def load(cls, path: Path) -> "CharNGramModel | None":
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            order=int(data["order"]),
            min_context=int(data.get("min_context", 8)),
            transitions={str(key): dict(value) for key, value in data["transitions"].items()},
            continuations={str(key): dict(value) for key, value in data.get("continuations", {}).items()},
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "type": "char_ngram_lm",
            "order": self.order,
            "min_context": self.min_context,
            "transitions": self.transitions,
            "continuations": self.continuations,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    def _next_char(self, context: str, prefer_terminator: bool = False) -> tuple[str, int] | None:
        max_length = min(self.order * 3, len(context))
        seen_keys: set[str] = set()
        for length in range(max_length, self.min_context - 1, -1):
            key = _key(context[-length:])
            if not key or len(key) > self.order or key in seen_keys:
                continue
            seen_keys.add(key)
            candidates = self.transitions.get(key)
            if not candidates:
                continue
            candidate_items = list(candidates.items())
            if context.rstrip()[-1:] in TERMINATORS:
                non_boundary = [item for item in candidate_items if item[0] not in TERMINATORS]
                if non_boundary:
                    candidate_items = non_boundary
            elif prefer_terminator:
                boundary = [item for item in candidate_items if item[0] in TERMINATORS]
                if boundary:
                    candidate_items = boundary
            char, _ = max(candidate_items, key=lambda item: (item[1], item[0]))
            return char, len(key)
        return None

    def _continuation_match(self, context: str) -> tuple[dict[str, int], int] | None:
        max_length = min(self.order * 3, len(context))
        seen_keys: set[str] = set()
        for length in range(max_length, self.min_context - 1, -1):
            key = context[-length:]
            if key in seen_keys:
                continue
            seen_keys.add(key)
            options = self.continuations.get(key)
            if options:
                return options, len(_key(key))
        return None

    def _continuation_options(self, context: str) -> dict[str, int] | None:
        match = self._continuation_match(context)
        return match[0] if match is not None else None

    def _build_normalized_continuation_buckets(self) -> dict[int, dict[str, Counter[str]]]:
        if self._normalized_continuation_buckets is not None:
            return self._normalized_continuation_buckets
        buckets: dict[int, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
        forms: dict[str, Counter[str]] = defaultdict(Counter)
        min_fuzzy_context = max(12, self.min_context)
        for raw_context, options in self.continuations.items():
            key = _key(raw_context)
            if min_fuzzy_context <= len(key) <= self.order:
                buckets[len(key)][key].update(options)
                forms[key][raw_context] += sum(options.values())
        self._normalized_continuation_buckets = {length: dict(values) for length, values in buckets.items()}
        self._normalized_context_forms = {key: counter for key, counter in forms.items()}
        return self._normalized_continuation_buckets

    @staticmethod
    def _edit_distance_at_most(left: str, right: str, limit: int) -> int | None:
        if abs(len(left) - len(right)) > limit:
            return None
        previous = list(range(len(right) + 1))
        for left_index, left_char in enumerate(left, start=1):
            current = [left_index]
            row_min = current[0]
            for right_index, right_char in enumerate(right, start=1):
                cost = 0 if left_char == right_char else 1
                value = min(
                    previous[right_index] + 1,
                    current[right_index - 1] + 1,
                    previous[right_index - 1] + cost,
                )
                current.append(value)
                row_min = min(row_min, value)
            if row_min > limit:
                return None
            previous = current
        distance = previous[-1]
        return distance if distance <= limit else None

    @staticmethod
    def _continuation_identity(text: str) -> str:
        return _key(text)

    @staticmethod
    def _key_positions(text: str) -> list[int]:
        positions: list[int] = []
        for index, char in enumerate(text):
            if _key(char):
                positions.append(index)
        return positions

    @classmethod
    def _correct_context_suffix(cls, context: str, corrected_suffix: str, matched_length: int) -> str | None:
        positions = cls._key_positions(context)
        corrected_key = _key(corrected_suffix)
        if len(positions) < matched_length or len(corrected_key) != matched_length:
            return None
        start = positions[-matched_length]
        end = positions[-1] + 1
        suffix_chars = list(context[start:end])
        key_index = 0
        for index, char in enumerate(suffix_chars):
            if _key(char):
                suffix_chars[index] = corrected_key[key_index]
                key_index += 1
        corrected = f"{context[:start]}{''.join(suffix_chars)}{context[end:]}"
        return corrected if corrected != context else None

    def _best_context_form(self, corrected_key: str | None) -> str | None:
        if corrected_key is None:
            return None
        if self._normalized_context_forms is None:
            self._build_normalized_continuation_buckets()
        assert self._normalized_context_forms is not None
        forms = self._normalized_context_forms.get(corrected_key)
        if not forms:
            return corrected_key
        return max(forms.items(), key=lambda item: (item[1], item[0]))[0]

    @staticmethod
    def _format_continuation(context: str, text: str) -> str:
        text = text.strip().strip(EDGE_QUOTES)
        if ends_with_separator(context):
            return strip_leading_separators(text).strip(EDGE_QUOTES)
        return text

    @staticmethod
    def _choose_formatted_continuation(options: Counter[str], context: str) -> dict[str, int]:
        if not options:
            return {}
        context_ends_at_boundary = ends_with_separator(context)

        def rank(item: tuple[str, int]) -> tuple[int, int, str]:
            text, count = item
            starts_at_boundary = starts_with_separator(text)
            boundary_score = int(not starts_at_boundary) if context_ends_at_boundary else int(starts_at_boundary)
            return count, boundary_score, text

        text, count = max(options.items(), key=rank)
        return {text: count}

    def verify(
        self,
        context: str,
        candidate_text: str,
        allow_fuzzy: bool = True,
        fuzzy_error_scale: float = 1.0,
    ) -> NGramVerification:
        context = context.strip()
        candidate_identity = self._continuation_identity(self._format_continuation(context, candidate_text))
        if not context or not candidate_identity:
            return NGramVerification(0.0, "ngram_empty_candidate")

        continuation_match = self._continuation_match(context)
        if continuation_match is not None:
            options, matched_length = continuation_match
            total = sum(options.values())
            for option, count in options.items():
                if self._continuation_identity(self._format_continuation(context, option)) == candidate_identity:
                    ambiguity_penalty = 0.10 if len(options) > 1 else 0.0
                    score = min(0.99, 0.62 + matched_length / max(1, self.order) * 0.22 + count / max(1, total) * 0.12)
                    return NGramVerification(max(0.0, score - ambiguity_penalty), "ngram_exact")
            return NGramVerification(0.0, "ngram_candidate_mismatch")

        if allow_fuzzy:
            fuzzy = self._fuzzy_continuation_options(context, fuzzy_error_scale=fuzzy_error_scale)
            if fuzzy is not None:
                options, matched_length, distance, corrected_key = fuzzy
                total = sum(options.values())
                for option, count in options.items():
                    if self._continuation_identity(self._format_continuation(context, option)) == candidate_identity:
                        corrected_suffix = self._best_context_form(corrected_key)
                        corrected_context = (
                            self._correct_context_suffix(context, corrected_suffix, matched_length)
                            if corrected_suffix
                            else None
                        )
                        score = min(
                            0.92,
                            0.50 + matched_length / max(1, self.order) * 0.26 + count / max(1, total) * 0.10 - distance * 0.04,
                        )
                        return NGramVerification(max(0.0, score), "ngram_fuzzy", corrected_context)

        return NGramVerification(0.0, "ngram_no_support")

    def _fuzzy_continuation_options(
        self,
        context: str,
        fuzzy_error_scale: float = 1.0,
    ) -> tuple[dict[str, int], int, int, str | None] | None:
        context_key = _key(context)
        min_fuzzy_context = max(12, self.min_context)
        if len(context_key) < min_fuzzy_context:
            return None
        buckets = self._build_normalized_continuation_buckets()
        best_distance: int | None = None
        best_length = 0
        best_key: str | None = None
        best_options: Counter[str] = Counter()

        for length in range(min(self.order, len(context_key)), min_fuzzy_context - 1, -1):
            suffix = context_key[-length:]
            max_distance = max(1, int((length // 5) * fuzzy_error_scale))
            matched_options: Counter[str] = Counter()
            matched_distance: int | None = None
            matched_keys: set[str] = set()
            for candidate_key, options in buckets.get(length, {}).items():
                if len(set(suffix) & set(candidate_key)) < length - max_distance * 2:
                    continue
                distance = self._edit_distance_at_most(suffix, candidate_key, max_distance)
                if distance is None:
                    continue
                if matched_distance is None or distance < matched_distance:
                    matched_distance = distance
                    matched_options = Counter(options)
                    matched_keys = {candidate_key}
                elif distance == matched_distance:
                    matched_options.update(options)
                    matched_keys.add(candidate_key)
            if matched_distance is None:
                continue
            if best_distance is None or matched_distance < best_distance or (
                matched_distance == best_distance and length > best_length
            ):
                best_distance = matched_distance
                best_length = length
                best_options = matched_options
                best_key = next(iter(matched_keys)) if len(matched_keys) == 1 else None

        if best_distance is None or not best_options:
            return None

        canonical: dict[str, Counter[str]] = defaultdict(Counter)
        for text, count in best_options.items():
            identity = self._continuation_identity(text)
            if identity:
                canonical[identity][self._format_continuation(context, text)] += count
        if len(canonical) != 1:
            return None
        formatted_options = next(iter(canonical.values()))
        return self._choose_formatted_continuation(formatted_options, context), best_length, best_distance, best_key

    def predict(
        self,
        context: str,
        max_chars: int = 80,
        allow_fuzzy: bool = True,
        fuzzy_error_scale: float = 1.0,
    ) -> NGramPrediction | None:
        context = context.strip()
        continuation_match = self._continuation_match(context)
        options = continuation_match[0] if continuation_match is not None else None
        continuation_matched_length = continuation_match[1] if continuation_match is not None else 0
        if options is not None and len(options) > 1:
            return None
        if options is not None and len(options) == 1:
            context_key_length = len(_key(context))
            if context_key_length >= 20 and continuation_matched_length < min(self.order, context_key_length):
                return None
            text, count = next(iter(options.items()))
            text = self._format_continuation(context, text)
            if text:
                confidence = min(0.99, 0.70 + min(count, 3) * 0.08)
                return NGramPrediction(text=text, confidence=confidence, reason="char_ngram")
        generated = []
        matched_lengths = []
        current = context
        pending_next = self._next_char(current)
        if pending_next is None and allow_fuzzy:
            fuzzy = self._fuzzy_continuation_options(context, fuzzy_error_scale=fuzzy_error_scale)
            if fuzzy is not None:
                options, matched_length, distance, corrected_key = fuzzy
                if len(options) == 1:
                    text = next(iter(options)).strip().strip(EDGE_QUOTES)
                    confidence = min(0.96, 0.55 + matched_length / max(1, self.order) * 0.35 - distance * 0.04)
                    corrected_suffix = self._best_context_form(corrected_key)
                    corrected_context = (
                        self._correct_context_suffix(context, corrected_suffix, matched_length)
                        if corrected_suffix
                        else None
                    )
                    return NGramPrediction(
                        text=text,
                        confidence=confidence,
                        reason="char_ngram_fuzzy",
                        corrected_context=corrected_context,
                    )
        if pending_next is not None and len(_key(context)) >= 20 and pending_next[1] < self.order:
            return None
        for _ in range(max_chars):
            next_char = pending_next or self._next_char(current, prefer_terminator=_has_cjk("".join(generated)))
            pending_next = None
            if next_char is None:
                return None
            char, matched_length = next_char
            generated.append(char)
            matched_lengths.append(matched_length)
            current += char
            if char in TERMINATORS and _has_cjk("".join(generated)):
                text = "".join(generated[:-1]).strip().strip(EDGE_QUOTES)
                confidence = min(0.99, 0.55 + min(matched_lengths) / max(1, self.order) * 0.4)
                return NGramPrediction(text=text, confidence=confidence, reason="char_ngram")
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Train an exported character n-gram LM for model-only fallback.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--order", type=int, default=32)
    parser.add_argument("--min-context", type=int, default=8)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    songs = _load_processed_songs(config.paths.processed_dir)
    if not songs:
        raise SystemExit("No processed songs found. Run prepare first.")
    model = CharNGramModel.train(songs, order=args.order, min_context=args.min_context)
    output = Path(args.output) if args.output else config.paths.model_dir / "ngram_model.json"
    model.save(output)
    print({"songs": len(songs), "order": args.order, "states": len(model.transitions), "output": str(output)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
