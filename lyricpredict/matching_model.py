from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .cleaner import CleanedSong
from .config import AppConfig, load_config
from .generation import Prediction
from .ngram_model import CharNGramModel
from .retrieval import LyricRetriever, TERMINATORS, _is_usable_line, _key, _load_processed_songs
from .separators import ends_with_separator, strip_leading_separators


STRICTNESS_THRESHOLDS = {
    "strict": 0.84,
    "balanced": 0.64,
    "tolerant": 0.44,
}

RETRIEVAL_THRESHOLDS = {
    "strict": 0.92,
    "balanced": 0.88,
    "tolerant": 0.72,
}

LEGACY_NGRAM_THRESHOLDS = {
    "strict": 0.82,
    "balanced": 0.62,
    "tolerant": 0.42,
}

CHAR_MATCH_AMBIGUITY_MARGINS = {
    "strict": 0.025,
    "balanced": 0.005,
    "tolerant": 0.001,
}


@dataclass(frozen=True)
class MatchingSample:
    context: str
    expected: str
    source: str
    kind: str
    index: int = -1


@dataclass(frozen=True)
class CandidateEntry:
    text: str
    source_context: str
    source: str
    kind: str
    index: int = -1

    @property
    def text_key(self) -> str:
        return _key(self.text)

    @property
    def context_key(self) -> str:
        return _key(self.source_context)


@dataclass(frozen=True)
class ScoredCandidate:
    candidate: CandidateEntry
    score: float
    reason: str


@dataclass(frozen=True)
class PairExample:
    context: str
    candidate: str
    label: int


@dataclass(frozen=True)
class MatchingTrainStats:
    variant: str
    positives: int
    examples: int
    epochs: int
    output: str
    elapsed_seconds: float


def normalize_strictness(strictness: str | None) -> str:
    value = (strictness or "balanced").replace("_", "-").lower()
    return value if value in STRICTNESS_THRESHOLDS else "balanced"


def cut_next_text(text: str) -> str:
    positions = [text.find(mark) for mark in TERMINATORS if text.find(mark) >= 0]
    if not positions:
        return text.strip()
    return text[: min(positions)].strip()


def format_expected(context: str, next_line: str) -> str:
    next_text = cut_next_text(next_line)
    if not next_text:
        return ""
    prefix = "" if context.rstrip().endswith(TERMINATORS) or next_text.startswith(TERMINATORS) else TERMINATORS[2]
    return f"{prefix}{next_text}"


def join_context_lines(lines: Iterable[str]) -> str:
    pieces: list[str] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if not pieces or pieces[-1].endswith(TERMINATORS) or line.startswith(TERMINATORS):
            pieces.append(line)
        else:
            pieces.append(f"{TERMINATORS[2]}{line}")
    return "".join(pieces)


def usable_song_lines(song: CleanedSong) -> list[str]:
    return [line.strip() for line in song.lines if _is_usable_line(line)]


def iter_adjacent_samples(songs: Sequence[CleanedSong]) -> Iterable[MatchingSample]:
    for song in songs:
        lines = usable_song_lines(song)
        for index in range(len(lines) - 1):
            expected = format_expected(lines[index], lines[index + 1])
            if expected:
                yield MatchingSample(lines[index], expected, song.source, "adjacent", index)


def split_semantic_parts(line: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    for char in line:
        if char in TERMINATORS:
            part = "".join(current).strip()
            if _is_usable_line(part):
                parts.append(part)
            current = []
            continue
        current.append(char)
    part = "".join(current).strip()
    if _is_usable_line(part):
        parts.append(part)
    return parts


def iter_half_sentence_samples(songs: Sequence[CleanedSong]) -> Iterable[MatchingSample]:
    for song in songs:
        lines = usable_song_lines(song)
        for line_index, line in enumerate(lines):
            for part_index, part in enumerate(split_semantic_parts(line)):
                key = _key(part)
                if len(key) < 8:
                    continue
                midpoint = max(2, len(part) // 2)
                context = part[:midpoint].strip()
                expected = part[midpoint:].strip()
                if _is_usable_line(context) and _is_usable_line(expected):
                    yield MatchingSample(context, expected, song.source, "half", line_index * 1000 + part_index)


def build_matching_samples(songs: Sequence[CleanedSong], include_half: bool = True) -> list[MatchingSample]:
    samples = list(iter_adjacent_samples(songs))
    if include_half:
        samples.extend(iter_half_sentence_samples(songs))
    return samples


def iter_window_candidates(songs: Sequence[CleanedSong], max_window_lines: int = 8) -> Iterable[CandidateEntry]:
    for song in songs:
        lines = usable_song_lines(song)
        for next_index in range(1, len(lines)):
            max_window = min(max_window_lines, next_index)
            for window in range(2, max_window + 1):
                start = next_index - window
                context = join_context_lines(lines[start:next_index])
                expected = format_expected(context, lines[next_index])
                if context and expected:
                    yield CandidateEntry(
                        text=expected,
                        source_context=context,
                        source=song.source,
                        kind=f"window-{window}",
                        index=next_index,
                    )


def build_candidate_library(
    songs: Sequence[CleanedSong],
    include_half: bool = True,
    include_windows: bool = True,
    max_window_lines: int = 8,
) -> list[CandidateEntry]:
    candidates: list[CandidateEntry] = []
    seen: set[tuple[str, str, str, int]] = set()
    sample_candidates = (
        CandidateEntry(sample.expected, sample.context, sample.source, sample.kind, sample.index)
        for sample in build_matching_samples(songs, include_half=include_half)
    )
    window_candidates = iter_window_candidates(songs, max_window_lines=max_window_lines) if include_windows else ()
    for candidate in list(sample_candidates) + list(window_candidates):
        identity = (_key(candidate.source_context), _key(candidate.text), candidate.source, candidate.index)
        if not identity[0] or not identity[1] or identity in seen:
            continue
        seen.add(identity)
        candidates.append(candidate)
    return candidates


def _edit_distance_limited(left: str, right: str, limit: int) -> int | None:
    if abs(len(left) - len(right)) > limit:
        return None
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
            return None
        previous = current
    distance = previous[-1]
    return distance if distance <= limit else None


def normalized_similarity(left: str, right: str) -> float:
    left_key = _key(left)
    right_key = _key(right)
    if not left_key or not right_key:
        return 0.0
    if left_key == right_key:
        return 1.0
    limit = max(1, max(len(left_key), len(right_key)) // 3)
    distance = _edit_distance_limited(left_key, right_key, limit)
    if distance is None:
        return 0.0
    return max(0.0, 1.0 - distance / max(len(left_key), len(right_key)))


def longest_common_substring_ratio(left: str, right: str) -> float:
    left_key = _key(left)
    right_key = _key(right)
    if not left_key or not right_key:
        return 0.0
    previous = [0] * (len(right_key) + 1)
    longest = 0
    for left_char in left_key:
        current = [0]
        for index, right_char in enumerate(right_key, start=1):
            value = previous[index - 1] + 1 if left_char == right_char else 0
            current.append(value)
            longest = max(longest, value)
        previous = current
    return longest / max(1, min(len(left_key), len(right_key)))


def char_ngrams(text: str, sizes: tuple[int, ...] = (2, 3, 4)) -> Counter[str]:
    key = _key(text)
    grams: Counter[str] = Counter()
    for size in sizes:
        if len(key) < size:
            continue
        grams.update(key[index : index + size] for index in range(len(key) - size + 1))
    if not grams and key:
        grams.update(key)
    return grams


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


class MatchingPredictor:
    name = "matching"

    def predict(
        self,
        context: str,
        strictness: str | None = None,
        exclude_sources: set[str] | None = None,
        correction: bool = False,
    ) -> Prediction:
        raise NotImplementedError


class LegacyNGramBenchmark(MatchingPredictor):
    name = "legacy-ngram-generator"

    def __init__(self, config: AppConfig):
        self.model = CharNGramModel.load(config.paths.model_dir / "ngram_model.json")

    def predict(
        self,
        context: str,
        strictness: str | None = None,
        exclude_sources: set[str] | None = None,
        correction: bool = False,
    ) -> Prediction:
        if not context.strip():
            return Prediction("", False, 0.0, "empty_context")
        if self.model is None:
            return Prediction("", False, 0.0, "missing_ngram_model")
        result = self.model.predict(context)
        if result is None:
            return Prediction("", False, 0.0, "legacy_ngram_no_match")
        threshold = LEGACY_NGRAM_THRESHOLDS[normalize_strictness(strictness)]
        accepted = result.confidence >= threshold
        return Prediction(
            result.text if accepted else "",
            accepted,
            result.confidence,
            result.reason if accepted else "legacy_ngram_threshold",
            result.corrected_context,
        )


class RetrievalBenchmark(MatchingPredictor):
    name = "retrieval"

    def __init__(self, config: AppConfig):
        root_dir = config.paths.processed_dir.parent.parent
        self.retriever = LyricRetriever(config.paths.processed_dir, extra_dirs=(root_dir / "selflyricdata",))

    def predict(
        self,
        context: str,
        strictness: str | None = None,
        exclude_sources: set[str] | None = None,
        correction: bool = False,
    ) -> Prediction:
        if not context.strip():
            return Prediction("", False, 0.0, "empty_context")
        result = self.retriever.find_next_line(context)
        if result is None:
            return Prediction("", False, 0.0, "retrieval_no_match")
        threshold = RETRIEVAL_THRESHOLDS[normalize_strictness(strictness)]
        accepted = result.confidence >= threshold
        return Prediction(
            result.text if accepted else "",
            accepted,
            result.confidence,
            result.reason if accepted else "retrieval_threshold",
            result.corrected_context,
        )


class CharMatchPredictor(MatchingPredictor):
    name = "char-match"

    def __init__(self, candidates: Sequence[CandidateEntry]):
        self.candidates = list(candidates)
        self.doc_grams = [char_ngrams(candidate.source_context) for candidate in self.candidates]
        df: Counter[str] = Counter()
        for grams in self.doc_grams:
            df.update(grams.keys())
        doc_count = max(1, len(self.doc_grams))
        self.idf = {gram: math.log((doc_count + 1) / (count + 0.5)) + 1.0 for gram, count in df.items()}
        inverted: dict[str, list[int]] = defaultdict(list)
        for index, grams in enumerate(self.doc_grams):
            for gram in grams:
                inverted[gram].append(index)
        self.inverted = dict(inverted)

    def _score(self, context: str, candidate: CandidateEntry, doc_grams: Counter[str]) -> ScoredCandidate:
        context_key = _key(context)
        candidate_key = candidate.context_key
        if not context_key or not candidate_key:
            return ScoredCandidate(candidate, 0.0, "char_match_empty")
        if candidate.kind == "half" and candidate_key.endswith(context_key) and len(context_key) >= 4:
            if candidate.source_context.strip() == context.strip():
                return ScoredCandidate(candidate, 0.999, "char_match_half_exact")
            coverage = min(1.0, len(context_key) / max(1, len(candidate_key)))
            return ScoredCandidate(candidate, 0.90 + 0.09 * coverage, "char_match_half_prefix")
        if context_key.endswith(candidate_key):
            coverage = min(1.0, len(candidate_key) / max(1, len(context_key)))
            return ScoredCandidate(candidate, 0.82 + 0.17 * coverage, "char_match_suffix")
        if candidate_key.endswith(context_key) and len(context_key) >= 4:
            coverage = min(1.0, len(context_key) / max(1, len(candidate_key)))
            return ScoredCandidate(candidate, 0.68 + 0.20 * coverage, "char_match_prefix")
        if candidate_key in context_key and not context_key.endswith(candidate_key):
            return ScoredCandidate(candidate, 0.40, "char_match_embedded_context")
        query = char_ngrams(context_key[-64:])
        if not query or not doc_grams:
            return ScoredCandidate(candidate, 0.0, "char_match_no_grams")
        total = sum(self.idf.get(gram, 1.0) * count for gram, count in query.items())
        overlap = 0.0
        for gram, count in query.items():
            if gram in doc_grams:
                overlap += self.idf.get(gram, 1.0) * min(count, doc_grams[gram])
        lexical = overlap / max(total, 1e-9)
        length_ratio = min(len(context_key), len(candidate_key)) / max(len(context_key), len(candidate_key), 1)
        score = min(0.99, lexical * 0.82 + length_ratio * 0.18)
        similarity = normalized_similarity(context_key, candidate_key)
        continuity = longest_common_substring_ratio(context_key, candidate_key)
        if similarity < 0.70 and continuity < 0.72:
            score = min(score, 0.40)
        return ScoredCandidate(candidate, score, "char_match_overlap")

    def rank(
        self,
        context: str,
        limit: int = 20,
        exclude_sources: set[str] | None = None,
    ) -> list[ScoredCandidate]:
        query = char_ngrams(_key(context)[-64:])
        candidate_scores: Counter[int] = Counter()
        for gram, count in query.items():
            weight = self.idf.get(gram, 1.0) * count
            for index in self.inverted.get(gram, ()):
                candidate_scores[index] += weight
        if candidate_scores:
            candidate_indexes = [
                index
                for index, _ in candidate_scores.most_common(max(200, limit * 20))
            ]
        else:
            candidate_indexes = list(range(len(self.candidates)))

        scored: list[ScoredCandidate] = []
        for index in candidate_indexes:
            candidate = self.candidates[index]
            if exclude_sources and candidate.source in exclude_sources:
                continue
            scored.append(self._score(context, candidate, self.doc_grams[index]))
        scored.sort(key=lambda item: (item.score, len(item.candidate.context_key)), reverse=True)
        return scored[:limit]

    @staticmethod
    def _format_output(context: str, text: str) -> str:
        return strip_leading_separators(text) if ends_with_separator(context) else text

    @staticmethod
    def _corrected_context(context: str, candidate: CandidateEntry) -> str | None:
        corrected = candidate.source_context.strip()
        original = context.strip()
        if not corrected or corrected == original:
            return None
        if normalized_similarity(original, corrected) < 0.70:
            return None
        return corrected

    def predict(
        self,
        context: str,
        strictness: str | None = None,
        exclude_sources: set[str] | None = None,
        correction: bool = False,
    ) -> Prediction:
        if not context.strip():
            return Prediction("", False, 0.0, "empty_context")
        ranked = self.rank(context, limit=3, exclude_sources=exclude_sources)
        if not ranked:
            return Prediction("", False, 0.0, "char_match_no_candidate")
        best = ranked[0]
        level = normalize_strictness(strictness)
        ambiguity_margin = CHAR_MATCH_AMBIGUITY_MARGINS[level]
        if len(ranked) > 1 and ranked[1].score >= best.score - ambiguity_margin and ranked[1].candidate.text_key != best.candidate.text_key:
            return Prediction("", False, best.score, "char_match_ambiguous")
        threshold = STRICTNESS_THRESHOLDS[level]
        accepted = best.score >= threshold
        text = self._format_output(context, best.candidate.text)
        corrected_context = self._corrected_context(context, best.candidate) if accepted and correction else None
        return Prediction(
            text if accepted else "",
            accepted,
            best.score,
            best.reason if accepted else "char_match_threshold",
            corrected_context,
        )


class LocalEncoder:
    def __init__(
        self,
        variant: str,
        token_to_id: dict[str, int] | None = None,
        tokenizer_name: str | None = None,
        hf_tokenizer=None,
    ):
        self.variant = variant
        self.token_to_id = token_to_id or {"<pad>": 0, "<unk>": 1, "<sep>": 2}
        self.tokenizer_name = tokenizer_name
        self.hf_tokenizer = hf_tokenizer

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def sep_id(self) -> int:
        return 2

    def raw_tokens(self, text: str) -> list[str]:
        if self.variant == "char-bigru":
            return list(_key(text))
        if self.hf_tokenizer is None:
            raise ValueError("token-bigru requires a tokenizer")
        ids = self.hf_tokenizer.encode(text, add_special_tokens=False)
        return [str(token_id) for token_id in ids]

    def fit(self, texts: Iterable[str], max_vocab_size: int = 8000) -> None:
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(self.raw_tokens(text))
        base = {"<pad>": 0, "<unk>": 1, "<sep>": 2}
        for token, _ in counter.most_common(max(0, max_vocab_size - len(base))):
            if token not in base:
                base[token] = len(base)
        self.token_to_id = base

    def encode_pair(self, context: str, candidate: str, max_context: int, max_candidate: int) -> list[int]:
        context_ids = [self.token_to_id.get(token, 1) for token in self.raw_tokens(context)][-max_context:]
        candidate_ids = [self.token_to_id.get(token, 1) for token in self.raw_tokens(candidate)][:max_candidate]
        return context_ids + [self.sep_id] + candidate_ids

    def to_json(self) -> dict:
        return {
            "variant": self.variant,
            "token_to_id": self.token_to_id,
            "tokenizer_name": self.tokenizer_name,
        }

    @classmethod
    def from_json(cls, data: dict):
        hf_tokenizer = None
        tokenizer_name = data.get("tokenizer_name")
        if data.get("variant") == "token-bigru":
            from transformers import AutoTokenizer

            hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
        return cls(
            variant=str(data["variant"]),
            token_to_id={str(key): int(value) for key, value in data["token_to_id"].items()},
            tokenizer_name=tokenizer_name,
            hf_tokenizer=hf_tokenizer,
        )


class PairDataset:
    def __init__(self, examples: Sequence[PairExample], encoder: LocalEncoder, max_context: int, max_candidate: int):
        self.examples = list(examples)
        self.encoder = encoder
        self.max_context = max_context
        self.max_candidate = max_candidate

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[list[int], float]:
        example = self.examples[index]
        return (
            self.encoder.encode_pair(example.context, example.candidate, self.max_context, self.max_candidate),
            float(example.label),
        )


def collate_pairs(batch: Sequence[tuple[list[int], float]], pad_id: int):
    import torch

    max_len = max(len(ids) for ids, _ in batch)
    input_ids = []
    mask = []
    labels = []
    for ids, label in batch:
        pad = max_len - len(ids)
        input_ids.append(torch.tensor(ids + [pad_id] * pad, dtype=torch.long))
        mask.append(torch.tensor([1] * len(ids) + [0] * pad, dtype=torch.float32))
        labels.append(label)
    return torch.stack(input_ids), torch.stack(mask), torch.tensor(labels, dtype=torch.float32)


def make_pair_ranker(vocab_size: int, embedding_dim: int = 96, hidden_dim: int = 128):
    import torch
    from torch import nn

    class PairRanker(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            self.gru = nn.GRU(
                embedding_dim,
                hidden_dim,
                batch_first=True,
                bidirectional=True,
            )
            self.head = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.10),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, input_ids, mask):
            embedded = self.embedding(input_ids)
            output, _ = self.gru(embedded)
            masked = output * mask.unsqueeze(-1)
            pooled = masked.sum(dim=1) / mask.sum(dim=1).clamp_min(1.0).unsqueeze(-1)
            return self.head(pooled).squeeze(-1)

    return PairRanker()


def load_torch_state_dict(path: Path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _same_source_negatives(sample: MatchingSample, samples: Sequence[MatchingSample], limit: int) -> list[str]:
    values = [
        other.expected
        for other in samples
        if other.source == sample.source and other.expected != sample.expected and other.context != sample.context
    ]
    return values[:limit]


def build_pair_examples(
    samples: Sequence[MatchingSample],
    candidates: Sequence[CandidateEntry],
    seed: int = 20260610,
    negatives_per_positive: int = 3,
) -> list[PairExample]:
    rng = random.Random(seed)
    all_texts = list({candidate.text for candidate in candidates if candidate.text_key})
    char_match = CharMatchPredictor(candidates)
    examples: list[PairExample] = []
    for sample in samples:
        examples.append(PairExample(sample.context, sample.expected, 1))
        negatives: list[str] = []
        negatives.extend(_same_source_negatives(sample, samples, limit=1))
        hard = [
            item.candidate.text
            for item in char_match.rank(sample.context, limit=12)
            if _key(item.candidate.text) != _key(sample.expected)
        ]
        negatives.extend(hard[:1])
        if all_texts:
            while len(negatives) < negatives_per_positive:
                candidate = rng.choice(all_texts)
                if _key(candidate) != _key(sample.expected):
                    negatives.append(candidate)
        seen: set[str] = set()
        for text in negatives:
            identity = _key(text)
            if not identity or identity in seen or identity == _key(sample.expected):
                continue
            seen.add(identity)
            examples.append(PairExample(sample.context, text, 0))
            if len(seen) >= negatives_per_positive:
                break
    rng.shuffle(examples)
    return examples


def _load_hf_tokenizer(tokenizer_name: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)


def _variant_stem(variant: str) -> str:
    return variant.replace("-", "_")


def train_bigru_ranker(
    config: AppConfig,
    variant: str,
    tokenizer_name: str | None = None,
    epochs: int = 8,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    seed: int = 20260610,
    max_pairs: int | None = None,
    output_dir: Path | None = None,
) -> MatchingTrainStats:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    if variant not in {"char-bigru", "token-bigru"}:
        raise ValueError(f"Unsupported matching variant: {variant}")

    started = time.perf_counter()
    random.seed(seed)
    torch.manual_seed(seed)
    songs = _load_processed_songs(config.paths.processed_dir)
    if not songs:
        raise ValueError("No processed songs found. Run prepare first.")
    samples = build_matching_samples(songs)
    if max_pairs is not None:
        samples = samples[:max_pairs]
    candidates = build_candidate_library(songs)
    examples = build_pair_examples(samples, candidates, seed=seed)

    hf_tokenizer = None
    if variant == "token-bigru":
        tokenizer_name = tokenizer_name or config.model.base_model
        hf_tokenizer = _load_hf_tokenizer(tokenizer_name)
    encoder = LocalEncoder(variant=variant, tokenizer_name=tokenizer_name, hf_tokenizer=hf_tokenizer)
    encoder.fit([value for example in examples for value in (example.context, example.candidate)])

    dataset = PairDataset(examples, encoder, max_context=96, max_candidate=48)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_pairs(batch, encoder.pad_id),
    )
    model = make_pair_ranker(len(encoder.token_to_id))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(max(1, epochs)):
        for input_ids, mask, labels in loader:
            optimizer.zero_grad()
            logits = model(input_ids, mask)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

    output_dir = output_dir or config.paths.model_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _variant_stem(variant)
    model_path = output_dir / f"matching_{stem}.pt"
    vocab_path = output_dir / f"matching_{stem}_vocab.json"
    config_path = output_dir / f"matching_{stem}_config.json"
    torch.save(model.state_dict(), model_path)
    vocab_path.write_text(json.dumps(encoder.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")
    config_path.write_text(
        json.dumps(
            {
                "variant": variant,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "seed": seed,
                "max_context": dataset.max_context,
                "max_candidate": dataset.max_candidate,
                "positives": len(samples),
                "examples": len(examples),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return MatchingTrainStats(
        variant=variant,
        positives=len(samples),
        examples=len(examples),
        epochs=epochs,
        output=str(model_path),
        elapsed_seconds=time.perf_counter() - started,
    )


class BigruRankerPredictor(MatchingPredictor):
    def __init__(
        self,
        config: AppConfig,
        variant: str,
        candidates: Sequence[CandidateEntry],
        model_dir: Path | None = None,
    ):
        import torch

        self.name = variant
        self.config = config
        self.variant = variant
        self.candidates = list(candidates)
        stem = _variant_stem(variant)
        artifact_dir = model_dir or config.paths.model_dir
        model_path = artifact_dir / f"matching_{stem}.pt"
        vocab_path = artifact_dir / f"matching_{stem}_vocab.json"
        config_path = artifact_dir / f"matching_{stem}_config.json"
        if not model_path.exists() or not vocab_path.exists() or not config_path.exists():
            raise FileNotFoundError(f"Missing trained matching model artifacts for {variant}. Run matching_model train first.")
        self.encoder = LocalEncoder.from_json(json.loads(vocab_path.read_text(encoding="utf-8")))
        self.model_config = json.loads(config_path.read_text(encoding="utf-8"))
        self.model = make_pair_ranker(len(self.encoder.token_to_id))
        self.model.load_state_dict(load_torch_state_dict(model_path))
        self.model.eval()
        self.prefilter = CharMatchPredictor(candidates)

    def rank(
        self,
        context: str,
        limit: int = 20,
        exclude_sources: set[str] | None = None,
    ) -> list[ScoredCandidate]:
        import torch

        prefiltered = self.prefilter.rank(context, limit=max(40, limit * 4), exclude_sources=exclude_sources)
        if not prefiltered:
            return []
        rows = [
            self.encoder.encode_pair(
                context,
                item.candidate.text,
                int(self.model_config.get("max_context", 96)),
                int(self.model_config.get("max_candidate", 48)),
            )
            for item in prefiltered
        ]
        batch = [(row, 0.0) for row in rows]
        input_ids, mask, _ = collate_pairs(batch, self.encoder.pad_id)
        with torch.no_grad():
            scores = torch.sigmoid(self.model(input_ids, mask)).detach().cpu().tolist()
        ranked = [
            ScoredCandidate(item.candidate, float(score), f"{self.variant}:ranker")
            for item, score in zip(prefiltered, scores)
        ]
        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked[:limit]

    def predict(
        self,
        context: str,
        strictness: str | None = None,
        exclude_sources: set[str] | None = None,
        correction: bool = False,
    ) -> Prediction:
        if not context.strip():
            return Prediction("", False, 0.0, "empty_context")
        ranked = self.rank(context, limit=3, exclude_sources=exclude_sources)
        if not ranked:
            return Prediction("", False, 0.0, f"{self.variant}_no_candidate")
        best = ranked[0]
        if len(ranked) > 1 and ranked[1].score >= best.score - 0.02 and ranked[1].candidate.text_key != best.candidate.text_key:
            return Prediction("", False, best.score, f"{self.variant}_ambiguous")
        threshold = STRICTNESS_THRESHOLDS[normalize_strictness(strictness)]
        accepted = best.score >= threshold
        return Prediction(
            best.candidate.text if accepted else "",
            accepted,
            best.score,
            best.reason if accepted else f"{self.variant}_threshold",
        )


def make_predictor(
    name: str,
    config: AppConfig,
    candidates: Sequence[CandidateEntry] | None = None,
    model_dirs: dict[str, Path] | None = None,
) -> MatchingPredictor:
    normalized = name.replace("_", "-").lower()
    if normalized == "legacy-ngram":
        normalized = "legacy-ngram-generator"
    if normalized == "legacy-ngram-generator":
        return LegacyNGramBenchmark(config)
    if normalized == "retrieval":
        return RetrievalBenchmark(config)
    candidates = list(candidates or build_candidate_library(_load_processed_songs(config.paths.processed_dir)))
    if normalized == "char-match":
        return CharMatchPredictor(candidates)
    if normalized in {"char-bigru", "token-bigru"}:
        model_dir = (model_dirs or {}).get(normalized)
        return BigruRankerPredictor(config, normalized, candidates, model_dir=model_dir)
    raise ValueError(f"Unsupported matching model: {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train or inspect experimental matching-style lyric models.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a BiGRU candidate ranker.")
    train_parser.add_argument("--config", default="configs/default.yaml")
    train_parser.add_argument("--variant", choices=("char-bigru", "token-bigru"), required=True)
    train_parser.add_argument("--tokenizer", default=None)
    train_parser.add_argument("--epochs", type=int, default=8)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--seed", type=int, default=20260610)
    train_parser.add_argument("--max-pairs", type=int, default=None)
    train_parser.add_argument("--output-dir", default=None)

    args = parser.parse_args()
    if args.command == "train":
        config = load_config(args.config)
        stats = train_bigru_ranker(
            config=config,
            variant=args.variant,
            tokenizer_name=args.tokenizer,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            max_pairs=args.max_pairs,
            output_dir=Path(args.output_dir) if args.output_dir else None,
        )
        print(json.dumps(asdict(stats), ensure_ascii=False, indent=2))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
