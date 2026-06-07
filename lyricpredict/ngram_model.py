from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from .cleaner import CleanedSong
from .config import load_config
from .retrieval import META_PREFIXES, TERMINATORS, _is_usable_line, _key, _load_processed_songs

EDGE_QUOTES = "\"'“”‘’"


@dataclass(frozen=True)
class NGramPrediction:
    text: str
    confidence: float
    reason: str


def _append_with_boundary(pieces: list[str], line: str) -> None:
    if not pieces:
        pieces.append(line)
        return
    separator = "" if pieces[-1].endswith(TERMINATORS) or line.startswith(TERMINATORS) else "，"
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
        if char in TERMINATORS and _has_cjk("".join(generated)):
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

    @classmethod
    def train(cls, songs: list[CleanedSong], order: int = 16, min_count: int = 1) -> "CharNGramModel":
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
        return cls(order=order, transitions=transitions, continuations=continuation_payload)

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

    def _next_char(self, context: str) -> tuple[str, int] | None:
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
            char, _ = max(candidate_items, key=lambda item: (item[1], item[0]))
            return char, len(key)
        return None

    def _continuation_options(self, context: str) -> dict[str, int] | None:
        max_length = min(self.order * 3, len(context))
        seen_keys: set[str] = set()
        for length in range(max_length, self.min_context - 1, -1):
            key = context[-length:]
            if key in seen_keys:
                continue
            seen_keys.add(key)
            options = self.continuations.get(key)
            if options:
                return options
        return None

    def predict(self, context: str, max_chars: int = 80) -> NGramPrediction | None:
        context = context.strip()
        options = self._continuation_options(context)
        if options is not None and len(options) > 1:
            return None
        generated = []
        matched_lengths = []
        current = context
        pending_next = self._next_char(current)
        if pending_next is not None and len(_key(context)) >= 20 and pending_next[1] < self.order:
            return None
        for _ in range(max_chars):
            next_char = pending_next or self._next_char(current)
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
    parser.add_argument("--order", type=int, default=16)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    songs = _load_processed_songs(config.paths.processed_dir)
    if not songs:
        raise SystemExit("No processed songs found. Run prepare first.")
    model = CharNGramModel.train(songs, order=args.order)
    output = Path(args.output) if args.output else config.paths.model_dir / "ngram_model.json"
    model.save(output)
    print({"songs": len(songs), "order": args.order, "states": len(model.transitions), "output": str(output)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
