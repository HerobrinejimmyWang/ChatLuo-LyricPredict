from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .cleaner import CleanedSong, clean_lyrics_file, clean_lyrics_text, decode_text

SUPPORTED_SUFFIXES = {".txt", ".lrc"}


@dataclass(frozen=True)
class ImportStats:
    files: int
    songs: int
    lines: int
    train_lines: int
    valid_lines: int


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


def prepare_dataset(raw_dir: Path, processed_dir: Path, validation_ratio: float = 0.1) -> ImportStats:
    processed_dir.mkdir(parents=True, exist_ok=True)
    songs = [clean_lyrics_file(path) for path in iter_lyric_files(raw_dir)]
    songs = [song for song in songs if song.lines]
    all_lines = [line for song in songs for line in song.lines]

    split_at = max(1, int(len(all_lines) * (1 - validation_ratio))) if all_lines else 0
    train_lines = all_lines[:split_at]
    valid_lines = all_lines[split_at:] if split_at < len(all_lines) else []

    songs_path = processed_dir / "songs.jsonl"
    with songs_path.open("w", encoding="utf-8") as handle:
        for song in songs:
            handle.write(json.dumps(asdict(song), ensure_ascii=False) + "\n")

    (processed_dir / "train.txt").write_text("\n".join(train_lines), encoding="utf-8")
    (processed_dir / "valid.txt").write_text("\n".join(valid_lines), encoding="utf-8")
    stats = ImportStats(
        files=len(list(iter_lyric_files(raw_dir))),
        songs=len(songs),
        lines=len(all_lines),
        train_lines=len(train_lines),
        valid_lines=len(valid_lines),
    )
    (processed_dir / "stats.json").write_text(json.dumps(asdict(stats), ensure_ascii=False, indent=2), encoding="utf-8")
    return stats
