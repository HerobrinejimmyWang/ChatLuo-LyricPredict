from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


LRC_TIME_RE = re.compile(r"\[\d{1,2}:\d{2}(?:[.:]\d{1,3})?\]")
LRC_META_RE = re.compile(r"^\[(?:ar|al|ti|au|by|offset|length|re|ve):.*\]$", re.IGNORECASE)
BRACKET_ONLY_RE = re.compile(r"^\[[^\]]+\]$")
SPACE_RE = re.compile(r"[ \t\u3000]+")
CREDIT_LINE_RE = re.compile(
    r"^\s*(?:"
    r"(?:中文)?VOCALOID(?:\s*Editor)?|Vocal(?:oid)?|PV|MV|"
    r"作词|作曲|编曲|调声|调教|调校|演唱|歌唱|主唱|人声|歌手|"
    r"吉他|贝斯|鼓|弦乐|钢琴|混音|母带|录音|和声|"
    r"制作人|制作|监制人|监制|音乐监制|出品人|出品|发行|版权|"
    r"曲绘|绘图|画师|封面|封面设计|动画|视频|影像|剪辑|后期|压制|渲染|"
    r"策划|协力|协助|特别感谢|"
    r"Lyricist|Lyrics?|Composer|Arrangement?|Arrange|Tuning|Singer|Vocaloid\s*Editor|"
    r"Guitar|Bass|Drums?|Piano|Mix(?:ing)?|Master(?:ing)?|Recording|Producer|"
    r"Executive\s+Producer|Music\s+Supervisor|Publisher|Copyright"
    r")(?:\b|(?=[\s/：:制作设计]))(?:\s*[^:：]+)?\s*[:：]",
    re.IGNORECASE,
)
INLINE_CREDIT_RE = re.compile(
    r"(?:VOCALOID|Vocal(?:oid)?|PV|MV|演唱|调校|调教|调声|封面|曲绘|动画|视频)\s*(?:[/A-Za-z\u4e00-\u9fff\s]*)?[:：]",
    re.IGNORECASE,
)
SHORT_CREDIT_RE = re.compile(r"(?:VOCALOID|资源组|制作组|字幕组|工作室|Studio)", re.IGNORECASE)


def is_credit_line(line: str) -> bool:
    normalized = normalize_line(line)
    if CREDIT_LINE_RE.match(normalized):
        return True
    if INLINE_CREDIT_RE.search(normalized):
        cjk = sum(1 for char in normalized if "\u4e00" <= char <= "\u9fff")
        return cjk <= 12
    if SHORT_CREDIT_RE.search(normalized):
        cjk = sum(1 for char in normalized if "\u4e00" <= char <= "\u9fff")
        return cjk <= 12
    return False


@dataclass(frozen=True)
class CleanedSong:
    source: str
    lines: list[str]

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


def decode_text(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def normalize_line(line: str) -> str:
    line = line.replace("\ufeff", "")
    line = line.replace("﹐", "，").replace("｡", "。")
    line = SPACE_RE.sub(" ", line).strip()
    return line


def clean_lrc_line(line: str) -> str | None:
    line = normalize_line(line)
    if not line:
        return None
    if LRC_META_RE.match(line):
        return None
    line = LRC_TIME_RE.sub("", line)
    line = normalize_line(line)
    if not line or BRACKET_ONLY_RE.match(line):
        return None
    if is_credit_line(line):
        return None
    return line


def clean_lyrics_text(text: str, source: str) -> CleanedSong:
    lines: list[str] = []
    for raw_line in text.splitlines():
        cleaned = clean_lrc_line(raw_line)
        if cleaned:
            lines.append(cleaned)
    return CleanedSong(source=source, lines=lines)


def clean_lyrics_file(path: Path) -> CleanedSong:
    return clean_lyrics_text(decode_text(path.read_bytes()), source=path.name)


def sentence_like_lines(song: CleanedSong) -> list[str]:
    return [line for line in song.lines if line.strip()]
