from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

from .cleaner import CleanedSong, clean_lyrics_file, clean_lyrics_text, decode_text
from .matching_model import MATCHING_INDEX_FILENAME, write_matching_index
from .retrieval import _key

SUPPORTED_SUFFIXES = {".txt", ".lrc"}
NEAR_DUPLICATE_CONTAINMENT_THRESHOLD = 0.85
NEAR_DUPLICATE_JACCARD_THRESHOLD = 0.78
NEAR_DUPLICATE_LINE_SIMILARITY = 0.84
NEAR_DUPLICATE_FUZZY_PREFILTER_CONTAINMENT = 0.45
NEAR_DUPLICATE_FUZZY_PREFILTER_JACCARD = 0.25
NEAR_DUPLICATE_FUZZY_PREFILTER_SHARED = 8


@dataclass(frozen=True)
class ImportStats:
    files: int
    songs: int
    lines: int
    train_lines: int
    valid_lines: int


@dataclass(frozen=True)
class SourceStats:
    source_dir: str
    files: int


@dataclass(frozen=True)
class SourceManifest:
    sources: list[SourceStats]
    stats: ImportStats
    duplicate_policy: str = "report-only"
    duplicate_report: str = "near_duplicate_report.json"
    matching_index: str = MATCHING_INDEX_FILENAME


@dataclass(frozen=True)
class ExactDuplicateEntry:
    kept_source: str
    skipped_source: str
    lines: int


@dataclass(frozen=True)
class NearDuplicatePair:
    source_a: str
    source_b: str
    containment: float
    jaccard: float
    shared_line_keys: int
    line_keys_a: int
    line_keys_b: int
    recommendation: str = "review"


@dataclass(frozen=True)
class DuplicateReport:
    exact_duplicates: list[ExactDuplicateEntry]
    near_duplicates: list[NearDuplicatePair]


def safe_filename(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._\-\u4e00-\u9fff]+", "_", Path(name).name).strip("._")
    return safe or "lyrics.txt"


def iter_lyric_files(raw_dir: Path) -> Iterable[Path]:
    if not raw_dir.exists():
        return []
    return sorted(path for path in raw_dir.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES)


def write_uploaded_file(raw_dir: Path, filename: str, data: bytes) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / safe_filename(filename)
    if target.exists():
        stem, suffix = target.stem, target.suffix
        index = 1
        while True:
            candidate = raw_dir / f"{stem}_{index}{suffix}"
            if not candidate.exists():
                target = candidate
                break
            index += 1
    target.write_bytes(data)
    return target


def clean_uploaded_bytes(filename: str, data: bytes) -> CleanedSong:
    return clean_lyrics_text(decode_text(data), source=safe_filename(filename))


def song_signature(song: CleanedSong) -> str:
    return "\n".join(line.strip().lower() for line in song.lines if line.strip())


def song_line_keys(song: CleanedSong) -> set[str]:
    return {key for key in (_key(line) for line in song.lines) if len(key) >= 2}


def _line_key_similarity(left: str, right: str) -> float:
    if left == right:
        return 1.0
    if len(left) < 4 or len(right) < 4:
        return 0.0
    return SequenceMatcher(None, left, right, autojunk=False).ratio()


def _shared_line_key_count(left_keys: set[str], right_keys: set[str]) -> int:
    exact = left_keys & right_keys
    shorter = sorted(left_keys - exact, key=len)
    longer = sorted(right_keys - exact, key=len)
    if len(left_keys) > len(right_keys):
        shorter = sorted(right_keys - exact, key=len)
        longer = sorted(left_keys - exact, key=len)
    used_longer: set[int] = set()
    fuzzy = 0
    for left in shorter:
        best_index = None
        best_score = 0.0
        for index, right in enumerate(longer):
            if index in used_longer:
                continue
            score = _line_key_similarity(left, right)
            if score > best_score:
                best_score = score
                best_index = index
        if best_index is not None and best_score >= NEAR_DUPLICATE_LINE_SIMILARITY:
            used_longer.add(best_index)
            fuzzy += 1
    return len(exact) + fuzzy


def find_near_duplicate_pairs(songs: list[CleanedSong]) -> list[NearDuplicatePair]:
    keyed = [(song, song_line_keys(song)) for song in songs]
    pairs: list[NearDuplicatePair] = []
    for left_index, (left_song, left_keys) in enumerate(keyed):
        if not left_keys:
            continue
        for right_song, right_keys in keyed[left_index + 1 :]:
            if not right_keys:
                continue
            exact_shared = len(left_keys & right_keys)
            if not exact_shared:
                continue
            exact_containment = exact_shared / max(1, min(len(left_keys), len(right_keys)))
            exact_jaccard = exact_shared / max(1, len(left_keys | right_keys))
            if exact_containment >= NEAR_DUPLICATE_CONTAINMENT_THRESHOLD or exact_jaccard >= NEAR_DUPLICATE_JACCARD_THRESHOLD:
                shared = exact_shared
            elif (
                exact_containment >= NEAR_DUPLICATE_FUZZY_PREFILTER_CONTAINMENT
                or exact_jaccard >= NEAR_DUPLICATE_FUZZY_PREFILTER_JACCARD
                or exact_shared >= NEAR_DUPLICATE_FUZZY_PREFILTER_SHARED
            ):
                shared = _shared_line_key_count(left_keys, right_keys)
            else:
                continue
            containment = shared / max(1, min(len(left_keys), len(right_keys)))
            jaccard = shared / max(1, len(left_keys | right_keys))
            if containment >= NEAR_DUPLICATE_CONTAINMENT_THRESHOLD or jaccard >= NEAR_DUPLICATE_JACCARD_THRESHOLD:
                pairs.append(
                    NearDuplicatePair(
                        source_a=left_song.source,
                        source_b=right_song.source,
                        containment=round(containment, 4),
                        jaccard=round(jaccard, 4),
                        shared_line_keys=shared,
                        line_keys_a=len(left_keys),
                        line_keys_b=len(right_keys),
                    )
                )
    return pairs


def deduplicate_exact_songs(songs: list[CleanedSong]) -> tuple[list[CleanedSong], list[ExactDuplicateEntry]]:
    deduped_songs: list[CleanedSong] = []
    seen_song_texts: dict[str, CleanedSong] = {}
    exact_duplicates: list[ExactDuplicateEntry] = []
    for song in songs:
        signature = song_signature(song)
        kept = seen_song_texts.get(signature)
        if kept is not None:
            exact_duplicates.append(
                ExactDuplicateEntry(
                    kept_source=kept.source,
                    skipped_source=song.source,
                    lines=len(song.lines),
                )
            )
            continue
        seen_song_texts[signature] = song
        deduped_songs.append(song)
    return deduped_songs, exact_duplicates


def write_duplicate_report(processed_dir: Path, exact_duplicates: list[ExactDuplicateEntry], songs: list[CleanedSong]) -> DuplicateReport:
    report = DuplicateReport(
        exact_duplicates=exact_duplicates,
        near_duplicates=find_near_duplicate_pairs(songs),
    )
    (processed_dir / "near_duplicate_report.json").write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def _prepare_dataset_from_files(
    files: list[Path],
    processed_dir: Path,
    validation_ratio: float = 0.1,
) -> ImportStats:
    processed_dir.mkdir(parents=True, exist_ok=True)
    songs = [clean_lyrics_file(path) for path in files]
    songs = [song for song in songs if song.lines]
    songs, exact_duplicates = deduplicate_exact_songs(songs)
    write_duplicate_report(processed_dir, exact_duplicates, songs)
    all_lines = [line for song in songs for line in song.lines]

    split_at = max(1, int(len(all_lines) * (1 - validation_ratio))) if all_lines else 0
    train_lines = all_lines[:split_at]
    valid_lines = all_lines[split_at:] if split_at < len(all_lines) else []

    songs_path = processed_dir / "songs.jsonl"
    with songs_path.open("w", encoding="utf-8") as handle:
        for song in songs:
            handle.write(json.dumps(asdict(song), ensure_ascii=False) + "\n")
    write_matching_index(processed_dir, songs)

    (processed_dir / "train.txt").write_text("\n".join(train_lines), encoding="utf-8")
    (processed_dir / "valid.txt").write_text("\n".join(valid_lines), encoding="utf-8")
    stats = ImportStats(
        files=len(files),
        songs=len(songs),
        lines=len(all_lines),
        train_lines=len(train_lines),
        valid_lines=len(valid_lines),
    )
    (processed_dir / "stats.json").write_text(json.dumps(asdict(stats), ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def prepare_dataset(raw_dir: Path, processed_dir: Path, validation_ratio: float = 0.1) -> ImportStats:
    return prepare_dataset_from_sources([raw_dir], processed_dir, validation_ratio)


def prepare_dataset_from_sources(
    raw_dirs: Iterable[Path],
    processed_dir: Path,
    validation_ratio: float = 0.1,
) -> ImportStats:
    sources: list[SourceStats] = []
    files: list[Path] = []
    for raw_dir in raw_dirs:
        source_files = list(iter_lyric_files(raw_dir))
        sources.append(SourceStats(source_dir=str(raw_dir), files=len(source_files)))
        files.extend(source_files)

    stats = _prepare_dataset_from_files(files, processed_dir, validation_ratio)
    manifest = SourceManifest(sources=sources, stats=stats)
    (processed_dir / "source_manifest.json").write_text(
        json.dumps(asdict(manifest), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return stats
